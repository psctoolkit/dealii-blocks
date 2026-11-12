// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from smoke-dealii, Copyright (C) 2026 Fabio Durastante.
// Matrices and Krylov vectors use the native deal.II backend wrappers.
#pragma once

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector_operation.h>

#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX)
#  define STOKES_HAS_PETSC
#  include <deal.II/lac/petsc_block_sparse_matrix.h>
#  include <deal.II/lac/petsc_block_vector.h>
#  include <deal.II/lac/petsc_sparse_matrix.h>
#  include <deal.II/lac/petsc_vector.h>
#  include <petscpc.h>
#endif
#if defined(DEAL_II_WITH_TRILINOS) && \
    ((defined(DEAL_II_TRILINOS_WITH_EPETRA) && defined(DEAL_II_TRILINOS_WITH_ML)) || \
     !DEAL_II_VERSION_GTE(9, 7, 0))
#  define STOKES_HAS_TRILINOS
#  include <deal.II/lac/trilinos_block_sparse_matrix.h>
#  include <deal.II/lac/trilinos_parallel_block_vector.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_vector.h>
#  include <deal.II/lac/trilinos_precondition.h>
#endif
#if defined(DEAL_II_WITH_PSBLAS) && defined(DEAL_II_WITH_AMG4PSBLAS)
#  define STOKES_HAS_PSBLAS
#  include <deal.II/lac/psblas_block_sparse_matrix.h>
#  include <deal.II/lac/psblas_block_vector.h>
#  include <deal.II/lac/psblas_sparse_matrix.h>
#  include <deal.II/lac/psblas_vector.h>
#  include <psb_base_cbind.h>
#  include <amg_c_dprec.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace StokesLA
{
  using size_type = dealii::types::global_dof_index;
  enum class Backend { petsc, trilinos, psctoolkit };

  inline void require(bool condition, const std::string &message)
  { AssertThrow(condition, dealii::ExcMessage(message)); }

  inline std::string backend_name(Backend backend)
  {
    switch (backend)
      {
        case Backend::petsc: return "PETSc";
        case Backend::trilinos: return "Trilinos";
        case Backend::psctoolkit: return "PSBLAS";
      }
    throw std::runtime_error("Unknown linear algebra backend");
  }

  inline std::string available_backends()
  {
    std::string result;
    const auto append = [&](const std::string &name) {
      if (!result.empty()) result += '|';
      result += name;
    };
    (void)append;
#ifdef STOKES_HAS_PSBLAS
    append("PSBLAS");
#endif
#ifdef STOKES_HAS_PETSC
    append("PETSc");
#endif
#ifdef STOKES_HAS_TRILINOS
    append("Trilinos");
#endif
    return result;
  }

  inline Backend default_backend()
  {
#ifdef STOKES_HAS_PSBLAS
    return Backend::psctoolkit;
#elif defined(STOKES_HAS_PETSC)
    return Backend::petsc;
#elif defined(STOKES_HAS_TRILINOS)
    return Backend::trilinos;
#else
    throw std::runtime_error("deal.II needs PETSc, Trilinos/Epetra or PSBLAS with AMG4PSBLAS");
#endif
  }

  inline Backend parse_backend(const std::string &name)
  {
#ifdef STOKES_HAS_PSBLAS
    if (name == "PSBLAS") return Backend::psctoolkit;
#endif
#ifdef STOKES_HAS_PETSC
    if (name == "PETSc") return Backend::petsc;
#endif
#ifdef STOKES_HAS_TRILINOS
    if (name == "Trilinos") return Backend::trilinos;
#endif
    throw std::runtime_error("Backend '" + name + "' is unavailable; this build supports " + available_backends());
  }

  /** Native names are forwarded unchanged. Empty entries retain library defaults. */
  struct NativePSCToolkitOptions
  {
    struct Entry { const char *name; const char *type; const char *description; };
    std::string ptype = "ML";
    std::map<std::string, std::string> values, pre, post, coarse;
    std::string additional_options, options_file, afmt, chfmt;
    bool output_details = false;
    static const std::vector<Entry> &catalogue()
    {
      static const std::vector<Entry> entries{
        {"MAX_LEVS","integer","Maximum hierarchy levels. Native ML uses at least two even when this is 1; AS, BJAC, etc. provide one-level actions."},
        {"MIN_COARSE_SIZE","integer","Global target coarse size."},
        {"MIN_COARSE_SIZE_PER_PROCESS","integer","Target coarse size per process."},
        {"MIN_CR_RATIO","real","Minimum aggregation ratio."},
        {"ML_CYCLE","string","Native multilevel cycle: VCYCLE, WCYCLE, KCYCLE, MULT or ADD."},
        {"OUTER_SWEEPS","integer","Multilevel cycles per application; native one-level actions use SMOOTHER_SWEEPS instead."},
        {"CYCLE_SWEEPS","integer","Native number of cycle sweeps at a level."},
        {"PAR_AGGR_ALG","string","Parallel aggregation algorithm: DECOUPLED, SYMDEC or COUPLED."},
        {"AGGR_TYPE","string","Native aggregation criterion, such as SOC1, SOC2 or MATCHBOXP."},
        {"AGGR_PROL","string","SMOOTHED or UNSMOOTHED prolongation."},
        {"AGGR_SIZE","integer","Matching aggregate size."},
        {"AGGR_ORD","string","Aggregation ordering, such as NATURAL or DEGREE."},
        {"AGGR_FILTER","string","FILTER or NOFILTER for prolongator construction."},
        {"AGGR_OMEGA_ALG","string","Native algorithm for prolongator damping."},
        {"AGGR_OMEGA_VAL","real","Prescribed prolongator damping value."},
        {"AGGR_EIG","string","Native spectral estimate for prolongator damping."},
        {"AGGR_THRESH","real","Strength-of-connection threshold. At the top level a list separated by commas, semicolons or spaces assigns levels 2,3,... as in the upstream example."},
        {"SMOOTHER_TYPE","string","Native smoother algorithm."},
        {"SMOOTHER_SWEEPS","integer","Smoother sweeps, including residual-correction sweeps for one-level JACOBI or L1-JACOBI."},
        {"SMOOTHER_STOP","string","Native residual stopping selection, TRUE or FALSE; FALSE keeps the requested sweep count fixed."},
        {"SMOOTHER_STOPTOL","real","Smoother residual stopping tolerance."},
        {"SMOOTHER_RESIDUAL","integer","Smoother residual check interval."},
        {"SMOOTHER_ITRACE","integer","Smoother trace frequency."},
        {"SMOOTHER_TRACE","string","Smoother trace selection."},
        {"SUB_SOLVE","string","Native local solver, including ILU, ILUT, MILU, MUMPS, INVK, INVT or AINV."},
        {"SUB_OVR","integer","Additive Schwarz overlap layers."},
        {"SUB_RESTR","string","Native Schwarz restriction: NONE or HALO."},
        {"SUB_PROL","string","Native Schwarz prolongation: NONE, SUM or AVG."},
        {"SOLVER_SWEEPS","integer","Inner solver sweeps."},
        {"SUB_FILLIN","integer","Local incomplete factorization fill."},
        {"SUB_ILUTHRS","real","Local ILUT dropping threshold."},
        {"INV_FILLIN","integer","Inverse factor fill."},
        {"INV_THRESH","real","Inverse factor dropping threshold."},
        {"AINV_ALG","string","AINV algorithm variant."},
        {"POLY_VARIANT","string","Native polynomial variant, such as CHEB_4 or CHEB_4_OPT."},
        {"POLY_DEGREE","integer","Polynomial degree."},
        {"POLY_RHO_ESTIMATE","string","Native spectral radius estimation method."},
        {"POLY_RHO_ESTIMATE_ITERATIONS","integer","Spectral estimation iteration count."},
        {"POLY_RHO_BA","real","Prescribed polynomial spectral radius; native admissible interval is (0,1]."},
        {"COARSE_SOLVE","string","Native coarse solver."},
        {"COARSE_SUBSOLVE","string","Native coarse local subsolver."},
        {"COARSE_MAT","string","Coarse matrix layout: DIST or REPL."},
        {"COARSE_SWEEPS","integer","Coarse smoother sweeps."},
        {"COARSE_FILLIN","integer","Coarse incomplete factorization fill."},
        {"COARSE_INVFILL","integer","Coarse inverse factor fill."},
        {"COARSE_ILUTHRS","real","Coarse ILUT dropping threshold."},
        {"KRM_METHOD","string","Native coarse Krylov method."},
        {"KRM_KPREC","string","Native coarse Krylov preconditioner."},
        {"KRM_SUB_SOLVE","string","Native coarse Krylov local solver."},
        {"KRM_GLOBAL","string","TRUE selects a global coarse Krylov solve."},
        {"KRM_EPS","real","Coarse Krylov stopping tolerance."},
        {"KRM_IRST","integer","Coarse Krylov restart dimension."},
        {"KRM_ISTOPC","integer","Coarse Krylov stopping criterion."},
        {"KRM_ITMAX","integer","Coarse Krylov iteration limit."},
        {"KRM_ITRACE","integer","Coarse Krylov trace frequency."},
        {"KRM_FILLIN","integer","Coarse Krylov local fill."},
        {"MUMPS_LOC_GLOB","string","MUMPS LOCAL_SOLVER or GLOBAL_SOLVER; indexed MUMPS entries use Additional options."},
        {"BJAC_STOP","string","Native coarse block-Jacobi residual stopping selection."},
        {"BJAC_STOPTOL","real","Native coarse block-Jacobi stopping tolerance."},
        {"BJAC_RESCHECK","integer","Native coarse block-Jacobi residual check interval."},
        {"BJAC_TRACE","string","Native coarse block-Jacobi trace selection."},
        {"BJAC_ITRACE","integer","Native coarse block-Jacobi trace frequency."}
      };
      return entries;
    }

    NativePSCToolkitOptions()
    {
      for (const auto &entry : catalogue())
        for (auto *scope : {&values, &pre, &post, &coarse})
          scope->emplace(entry.name, "");
    }

    void declare_parameters(dealii::ParameterHandler &prm)
    {
      prm.add_parameter("ptype", ptype,
        "Original amg_c_dprecinit ptype: ML, AS, BJAC, JACOBI, L1-JACOBI, POLY, NONE, etc. No presets or aliases are substituted.");
      const auto declare_scope = [&](std::map<std::string, std::string> &scope) {
        for (const auto &entry : catalogue())
          prm.add_parameter(entry.name, scope.at(entry.name),
            std::string(entry.description) + " Empty retains the AMG4PSBLAS default.");
      };
      declare_scope(values);
      for (const auto &scope : std::vector<std::pair<std::string, std::map<std::string, std::string> *>>{
             {"PRE", &pre}, {"POST", &post}, {"COARSE", &coarse}})
        {
          prm.enter_subsection(scope.first);
          declare_scope(*scope.second);
          prm.leave_subsection();
        }
      prm.add_parameter("Additional options", additional_options,
        "Native typed entries separated by semicolons: hierarchy|smoother integer|real|string NAME VALUE [level=N|N:M|coarse] [position=PRE|POST] [index=N]. Values containing spaces must be quoted. These entries follow the named entries.");
      prm.add_parameter("Options file", options_file,
        "Optional native typed entries, one per line; applied after Additional options. Relative paths are resolved against the main parameter file.");
      prm.add_parameter("afmt", afmt, "Native matrix format argument; empty preserves the native default.");
      prm.add_parameter("chfmt", chfmt, "Native vector format argument; empty preserves the native default.");
      prm.add_parameter("Verbosity", output_details, "Print amg_c_ddescr after setup.");
    }

    void resolve_paths(const std::filesystem::path &directory)
    {
      if (!options_file.empty() && std::filesystem::path(options_file).is_relative())
        options_file = (directory / options_file).lexically_normal().string();
    }
  };

  struct PreconditionerSettings
  {
    explicit PreconditionerSettings(bool pressure = false)
      : petsc_pc_type(pressure ? "jacobi" : "gamg"),
        trilinos_type(pressure ? "Jacobi" : "AMG")
    {
      psc.ptype = pressure ? "JACOBI" : "ML";
      if (pressure) smoother_sweeps = 1;
    }

    NativePSCToolkitOptions psc;
    std::string petsc_pc_type, petsc_options;
    std::string trilinos_type;
    std::string smoother_type = "symmetric Gauss-Seidel", coarse_type = "Amesos-KLU";
    unsigned int smoother_sweeps = 2, smoother_overlap = 0, ilu_fill = 0, n_cycles = 1;
    double aggregation_threshold = 1e-4, relaxation = 1., ilu_atol = 0., ilu_rtol = 1., ilut_drop = 1e-3;
    bool w_cycle = false, higher_order_elements = true, output_details = false;

    void declare_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("AMG4PSBLAS");
      psc.declare_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("PETSc");
      prm.add_parameter("pc_type", petsc_pc_type,
        "Native PETSc PC type, for example gamg, hypre, asm, bjacobi, jacobi, ilu, lu or none. Availability follows the PETSc build.");
      prm.add_parameter("Options", petsc_options,
        "Native PETSc options, e.g. -pc_asm_overlap 1 -sub_pc_type ilu. This private PC options database does not modify other blocks.");
      prm.leave_subsection();
      prm.enter_subsection("Trilinos");
      prm.add_parameter("Preconditioner", trilinos_type,
        "deal.II Trilinos preconditioner: AMG, Jacobi, SSOR, ILU, ILUT, IC or Identity.",
        dealii::Patterns::Selection("AMG|Jacobi|SSOR|ILU|ILUT|IC|Identity"));
      prm.add_parameter("smoother_type", smoother_type, "ML smoother selection.");
      prm.add_parameter("coarse_type", coarse_type, "ML coarse solver selection.");
      prm.add_parameter("smoother_sweeps", smoother_sweeps, "AMG or relaxation sweeps.", dealii::Patterns::Integer(1));
      prm.add_parameter("smoother_overlap", smoother_overlap, "Ifpack or AMG overlap.");
      prm.add_parameter("n_cycles", n_cycles, "ML cycles per application.", dealii::Patterns::Integer(1));
      prm.add_parameter("w_cycle", w_cycle, "Use ML W cycles.");
      prm.add_parameter("aggregation_threshold", aggregation_threshold, "ML aggregation threshold.", dealii::Patterns::Double(0.));
      prm.add_parameter("higher_order_elements", higher_order_elements, "ML higher order finite-element flag.");
      prm.add_parameter("relaxation", relaxation, "Jacobi or SSOR damping.", dealii::Patterns::Double(0.));
      prm.add_parameter("ilu_fill", ilu_fill, "ILU, ILUT or IC fill level.");
      prm.add_parameter("ilu_atol", ilu_atol, "Incomplete factorization diagonal absolute perturbation.", dealii::Patterns::Double(0.));
      prm.add_parameter("ilu_rtol", ilu_rtol, "Incomplete factorization diagonal relative perturbation.", dealii::Patterns::Double(0.));
      prm.add_parameter("ilut_drop", ilut_drop, "ILUT dropping threshold.", dealii::Patterns::Double(0.));
      prm.add_parameter("output_details", output_details, "Print ML hierarchy details.");
      prm.leave_subsection();
    }

    void validate() const
    {
      require(!psc.ptype.empty() && psc.ptype.size() <= 80,
              "AMG4PSBLAS/ptype must contain 1 to 80 characters");
      require(psc.afmt.size() <= 10 && psc.chfmt.size() <= 6,
              "AMG4PSBLAS afmt/chfmt exceed their native string bounds");
      require(!petsc_pc_type.empty(), "PETSc/pc_type cannot be empty");
    }
  };

  /** A type-erased preconditioner application without changing vector storage. */
  template <typename VectorType>
  class Preconditioner
  {
  public:
    virtual ~Preconditioner() = default;
    virtual void vmult(VectorType &dst, const VectorType &src) const = 0;
  };

  template <typename VectorType>
  class FunctionPreconditioner final : public Preconditioner<VectorType>
  {
  public:
    explicit FunctionPreconditioner(std::function<void(VectorType &, const VectorType &)> action)
      : action(std::move(action)) {}
    void vmult(VectorType &dst, const VectorType &src) const override { action(dst, src); }
  private:
    std::function<void(VectorType &, const VectorType &)> action;
  };

  namespace detail
  {
    template <typename VectorType, typename NativePreconditioner>
    std::unique_ptr<Preconditioner<VectorType>>
    wrap_native_preconditioner(std::shared_ptr<NativePreconditioner> pc)
    {
      return std::make_unique<FunctionPreconditioner<VectorType>>(
        [pc](VectorType &dst, const VectorType &src) { pc->vmult(dst, src); });
    }

    /** Shared initialization signatures of the native deal.II wrappers. */
    template <typename MatrixType, typename BlockMatrixType,
              typename VectorType, typename BlockVectorType>
    struct NativeBackendTypes
    {
      using Matrix = MatrixType;
      using BlockMatrix = BlockMatrixType;
      using Vector = VectorType;
      using BlockVector = BlockVectorType;

      static void reinit_matrix(Matrix &matrix, const dealii::IndexSet &rows,
                                const dealii::IndexSet &columns,
                                const dealii::DynamicSparsityPattern &graph,
                                MPI_Comm comm)
      { matrix.reinit(rows, columns, graph, comm); }

      static void reinit_block_matrix(BlockMatrix &matrix,
                                      const std::vector<dealii::IndexSet> &owned,
                                      const dealii::BlockDynamicSparsityPattern &graph,
                                      MPI_Comm comm)
      { matrix.reinit(owned, graph, comm); }

      static void reinit_block_vector(BlockVector &vector,
                                      const std::vector<dealii::IndexSet> &owned,
                                      MPI_Comm comm)
      { vector.reinit(owned, comm); }

      static void reinit_ghosted_block_vector(BlockVector &vector,
                                              const std::vector<dealii::IndexSet> &owned,
                                              const std::vector<dealii::IndexSet> &relevant,
                                              MPI_Comm comm)
      { vector.reinit(owned, relevant, comm); }

      static void import_ghosts(BlockVector &destination, const BlockVector &source)
      {
        // Native assignment imports the owned entries and updates the ghost
        // values. This is used for finite-element evaluation and output, not
        // inside matrix products or preconditioner applications.
        destination = source;
      }
    };

#ifdef STOKES_HAS_PETSC
    inline void petsc_check(PetscErrorCode status, const char *operation)
    { require(status == 0, std::string(operation) + " failed with PETSc error " + std::to_string(status)); }

    struct PetscPC
    {
      ~PetscPC()
      {
        if (pc) (void)PCDestroy(&pc);
        if (options) (void)PetscOptionsDestroy(&options);
      }
      PC pc = nullptr;
      PetscOptions options = nullptr;
    };
#endif

#ifdef STOKES_HAS_PSBLAS
    inline void psc_check(psb_i_t status, const std::string &operation)
    { require(status == 0, operation + " failed with PSBLAS/AMG4PSBLAS error " + std::to_string(status)); }

    struct NativeOption
    {
      std::string phase, type, name, value, position;
      psb_i_t first = 0, last = 0, index = 0;
      bool coarse = false;
    };

    inline psb_i_t native_integer(const std::string &text)
    {
      std::size_t used = 0;
      const auto value = std::stoll(text, &used);
      require(used == text.size() && value >= std::numeric_limits<psb_i_t>::min() &&
              value <= std::numeric_limits<psb_i_t>::max(), "Invalid native integer: " + text);
      return static_cast<psb_i_t>(value);
    }

    inline double native_real(std::string text)
    {
      std::replace(text.begin(), text.end(), 'D', 'E');
      std::replace(text.begin(), text.end(), 'd', 'e');
      std::size_t used = 0;
      const auto value = std::stod(text, &used);
      require(used == text.size() && std::isfinite(value), "Invalid native real: " + text);
      return value;
    }

    inline std::vector<NativeOption> parse_native_options(std::string text)
    {
      bool quoted = false, escaped = false;
      for (auto &c : text)
        {
          if (!escaped && c == '"') quoted = !quoted;
          if (!quoted && c == ';') c = '\n';
          escaped = !escaped && c == '\\';
        }
      require(!quoted, "Unterminated quoted native option value");
      std::vector<NativeOption> result;
      std::istringstream lines(text);
      for (std::string line; std::getline(lines, line); )
        {
          std::istringstream fields(line);
          fields >> std::ws;
          if (fields.eof() || fields.peek() == '#') continue;
          NativeOption option;
          require(static_cast<bool>(fields >> option.phase >> option.type >> option.name >> std::quoted(option.value)),
            "Expected: hierarchy|smoother integer|real|string NAME VALUE [level=N|N:M|coarse] [position=PRE|POST] [index=N]");
          require(option.phase == "hierarchy" || option.phase == "smoother", "Invalid native setup phase: " + option.phase);
          require(option.type == "integer" || option.type == "real" || option.type == "string", "Invalid native option type: " + option.type);
          for (std::string selector; fields >> selector; )
            {
              if (selector[0] == '#') break;
              const auto separator = selector.find('=');
              require(separator != std::string::npos, "Invalid native selector: " + selector);
              const auto key = selector.substr(0, separator), value = selector.substr(separator + 1);
              if (key == "level")
                {
                  if (value == "coarse") option.coarse = true;
                  else
                    {
                      const auto colon = value.find(':');
                      option.first = native_integer(value.substr(0, colon));
                      option.last = colon == std::string::npos ? option.first : native_integer(value.substr(colon + 1));
                      require(option.first > 0 && option.last >= option.first, "Invalid native level range: " + value);
                    }
                }
              else if (key == "position")
                {
                  require(value == "PRE" || value == "POST", "Native position must be PRE or POST");
                  option.position = value;
                }
              else if (key == "index")
                {
                  option.index = native_integer(value);
                  require(option.index > 0, "Native indexed options are one-based");
                }
              else throw std::runtime_error("Unknown native selector: " + key);
            }
          require(!option.name.empty() && option.name.size() <= 80 && option.value.size() <= 80,
                  "Native parameter names and string values have an 80-character limit");
          if (option.type == "integer") (void)native_integer(option.value);
          if (option.type == "real") (void)native_real(option.value);
          result.push_back(std::move(option));
        }
      return result;
    }

    inline std::vector<NativeOption> native_options(const NativePSCToolkitOptions &settings, MPI_Comm comm)
    {
      std::vector<NativeOption> result;
      const auto append = [&](const std::map<std::string, std::string> &scope,
                              const std::string &position, bool coarse) {
        for (const auto &entry : NativePSCToolkitOptions::catalogue())
          {
            const auto &value = scope.at(entry.name);
            if (value.empty()) continue;
            if (std::string(entry.name) == "AGGR_THRESH" && position.empty() && !coarse)
              {
                std::string list = value;
                for (auto &c : list) if (c == ',' || c == ';') c = ' ';
                std::istringstream values(list);
                std::vector<std::string> thresholds;
                for (std::string v; values >> v; ) thresholds.push_back(v);
                require(!thresholds.empty(), "AGGR_THRESH needs at least one real value");
                if (thresholds.size() > 1)
                  {
                    psb_i_t level = 2;
                    for (const auto &v : thresholds)
                      {
                        result.push_back({"hierarchy", "real", entry.name, v, "", level, level, 0, false});
                        ++level;
                      }
                    continue;
                  }
              }
            NativeOption option;
            option.phase = position.empty() && !coarse ? "hierarchy" : "smoother";
            option.name = entry.name; option.type = entry.type; option.value = value;
            option.position = position; option.coarse = coarse;
            // Native BJAC_* settings target the coarse block-Jacobi solver.
            if (option.name.rfind("BJAC_", 0) == 0) option.coarse = true;
            result.push_back(std::move(option));
          }
      };
      append(settings.values, "", false);
      append(settings.pre, "PRE", false);
      append(settings.post, "POST", false);
      append(settings.coarse, "", true);
      const auto extra = parse_native_options(settings.additional_options);
      result.insert(result.end(), extra.begin(), extra.end());
      if (!settings.options_file.empty())
        {
          std::string contents, error;
          if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
            {
              std::ifstream file(settings.options_file);
              if (!file) error = "Cannot open AMG4PSBLAS options file: " + settings.options_file;
              else contents.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }
          error = dealii::Utilities::MPI::broadcast(comm, error);
          require(error.empty(), error);
          contents = dealii::Utilities::MPI::broadcast(comm, contents);
          const auto file_options = parse_native_options(contents);
          result.insert(result.end(), file_options.begin(), file_options.end());
        }
      return result;
    }

    inline psb_i_t native_levels(amg_c_dprec *preconditioner)
    {
      amg_c_prec_level_info info{};
      psb_i_t level = 1;
      psc_check(amg_c_dprec_get_level_info(preconditioner, level, &info), "amg_c_dprec_get_level_info");
      while (level < std::numeric_limits<psb_i_t>::max() &&
             amg_c_dprec_get_level_info(preconditioner, level + 1, &info) == 0) ++level;
      return level;
    }

    inline void apply_native_options(amg_c_dprec *pc, const std::vector<NativeOption> &options,
                                     const std::string &phase)
    {
      for (const auto &option : options)
        if (option.phase == phase)
          {
            require(!option.name.empty() && option.name.size() <= 80 &&
                    (option.type != "string" || option.value.size() <= 80),
                    "Native parameter names and string values have an 80-character limit");
            psb_i_t first = option.first, last = option.last;
            if (option.coarse) first = last = native_levels(pc);
            if (option.type == "string" && first > 0)
              {
                // These limitations belong to the pinned optional string
                // setter. Reject them instead of silently changing a different level.
                const auto levels = native_levels(pc);
                const bool only_coarse = first == levels && last == levels;
                const bool coarse_routing = option.name == "SUB_SOLVE" || option.name == "ML_CYCLE" ||
                  option.name == "PAR_AGGR_ALG" || option.name == "AGGR_TYPE" ||
                  option.name == "SUB_RESTR" || option.name == "SUB_PROL";
                require(!coarse_routing || only_coarse,
                  "This pinned AMG4PSBLAS optional setter routes level-specific " + option.name +
                  " to the coarse level; use an unqualified entry or PRE/POST instead");
                if (option.name == "SMOOTHER_TYPE")
                  {
                    std::string value = option.value;
                    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::toupper(c); });
                    const bool unsupported = value == "POLY" || value == "AS" || value == "NONE" ||
                      value == "NOPREC" || value == "JAC" || value == "L1-BWGS";
                    const bool coarse_only = value == "JACOBI" || value == "L1-JACOBI" ||
                      value == "L1-GS" || value == "L1-FWGS" || value == "L1-FBGS";
                    require(!unsupported && (!coarse_only || only_coarse),
                      "This pinned AMG4PSBLAS cannot select " + value +
                      " through that level-specific SMOOTHER_TYPE call; use native ptype or PRE/POST");
                  }
                require(option.name != "COARSE_MAT" || option.value != "DIST",
                  "The pinned level-specific COARSE_MAT setter uses the native value DISTR; the unqualified setter also accepts DIST");
              }
            const auto *begin = first ? &first : nullptr;
            const auto *end = last ? &last : nullptr;
            const auto *index = option.index ? &option.index : nullptr;
            const char *position = option.position.empty() ? nullptr : option.position.c_str();
            psb_i_t status = 0;
            if (option.type == "integer")
              status = amg_c_dprecseti_opt(pc, option.name.c_str(), native_integer(option.value), begin, end, position, index);
            else if (option.type == "real")
              status = amg_c_dprecsetr_opt(pc, option.name.c_str(), native_real(option.value), begin, end, position, index);
            else
              status = amg_c_dprecsetc_opt(pc, option.name.c_str(), option.value.c_str(), begin, end, position, index);
            psc_check(status, "AMG4PSBLAS set " + option.name + "=" + option.value);
          }
    }

    /** Own only the native AMG hierarchy; the matrix outlives this object. */
    struct PscAMG
    {
      PscAMG() : pc(amg_c_dprec_new())
      { require(pc != nullptr, "AMG4PSBLAS preconditioner allocation failed"); }
      ~PscAMG()
      {
        if (pc != nullptr)
          {
            if (pc->dprec != nullptr) (void)amg_c_dprec_delete(pc);
            else std::free(pc);
          }
      }
      PscAMG(const PscAMG &) = delete;
      PscAMG &operator=(const PscAMG &) = delete;
      amg_c_dprec *pc;
    };
#endif
  } // namespace detail

  template <Backend backend>
  struct BackendTraits;

#ifdef STOKES_HAS_PETSC
  template <>
  struct BackendTraits<Backend::petsc>
    : detail::NativeBackendTypes<dealii::PETScWrappers::MPI::SparseMatrix,
                                 dealii::PETScWrappers::MPI::BlockSparseMatrix,
                                 dealii::PETScWrappers::MPI::Vector,
                                 dealii::PETScWrappers::MPI::BlockVector>
  {
    static std::unique_ptr<Preconditioner<Vector>>
    make_preconditioner(const Matrix &matrix, const PreconditionerSettings &settings)
    {
      auto state = std::make_shared<detail::PetscPC>();
      detail::petsc_check(PCCreate(matrix.get_mpi_communicator(), &state->pc), "PCCreate");
      detail::petsc_check(PCSetOperators(state->pc, static_cast<Mat>(matrix), static_cast<Mat>(matrix)), "PCSetOperators");
      detail::petsc_check(PCSetType(state->pc, settings.petsc_pc_type.c_str()), "PCSetType");
      detail::petsc_check(PetscOptionsCreate(&state->options), "PetscOptionsCreate");
      detail::petsc_check(PetscOptionsInsertString(state->options, settings.petsc_options.c_str()), "PetscOptionsInsertString");
      detail::petsc_check(PetscObjectSetOptions(reinterpret_cast<PetscObject>(state->pc), state->options), "PetscObjectSetOptions");
      detail::petsc_check(PCSetFromOptions(state->pc), "PCSetFromOptions");
      detail::petsc_check(PCSetUp(state->pc), "PCSetUp");
      return std::make_unique<FunctionPreconditioner<Vector>>(
        [state](Vector &dst, const Vector &src) {
          detail::petsc_check(PCApply(state->pc, static_cast<Vec>(src), static_cast<Vec>(dst)), "PCApply");
        });
    }
  };
#endif

#ifdef STOKES_HAS_TRILINOS
  template <>
  struct BackendTraits<Backend::trilinos>
    : detail::NativeBackendTypes<dealii::TrilinosWrappers::SparseMatrix,
                                 dealii::TrilinosWrappers::BlockSparseMatrix,
                                 dealii::TrilinosWrappers::MPI::Vector,
                                 dealii::TrilinosWrappers::MPI::BlockVector>
  {
    static std::unique_ptr<Preconditioner<Vector>>
    make_preconditioner(const Matrix &matrix, const PreconditionerSettings &s)
    {
      using namespace dealii::TrilinosWrappers;
        if (s.trilinos_type == "Identity")
          return std::make_unique<FunctionPreconditioner<Vector>>([](Vector &dst, const Vector &src) { dst = src; });
        if (s.trilinos_type == "AMG")
          {
#if defined(DEAL_II_TRILINOS_WITH_ML) || !DEAL_II_VERSION_GTE(9, 7, 0)
            auto pc = std::make_shared<PreconditionAMG>();
            PreconditionAMG::AdditionalData data;
            data.elliptic = true;
            data.higher_order_elements = s.higher_order_elements;
            data.n_cycles = s.n_cycles;
            data.w_cycle = s.w_cycle;
            data.aggregation_threshold = s.aggregation_threshold;
            data.smoother_sweeps = s.smoother_sweeps;
            data.smoother_overlap = s.smoother_overlap;
            data.output_details = s.output_details;
            data.smoother_type = s.smoother_type.c_str();
            data.coarse_type = s.coarse_type.c_str();
            pc->initialize(matrix, data);
            return detail::wrap_native_preconditioner<Vector>(pc);
#else
            throw std::runtime_error("Trilinos AMG requires ML in the deal.II build");
#endif
          }
        if (s.trilinos_type == "Jacobi")
          {
            auto pc = std::make_shared<PreconditionJacobi>();
            pc->initialize(matrix, PreconditionJacobi::AdditionalData(s.relaxation, 0., s.smoother_sweeps));
            return detail::wrap_native_preconditioner<Vector>(pc);
          }
        if (s.trilinos_type == "SSOR")
          {
            auto pc = std::make_shared<PreconditionSSOR>();
            pc->initialize(matrix, PreconditionSSOR::AdditionalData(s.relaxation, 0., s.smoother_overlap, s.smoother_sweeps));
            return detail::wrap_native_preconditioner<Vector>(pc);
          }
        if (s.trilinos_type == "ILU")
          {
            auto pc = std::make_shared<PreconditionILU>();
            pc->initialize(matrix, PreconditionILU::AdditionalData(s.ilu_fill, s.ilu_atol, s.ilu_rtol, s.smoother_overlap));
            return detail::wrap_native_preconditioner<Vector>(pc);
          }
        if (s.trilinos_type == "ILUT")
          {
            auto pc = std::make_shared<PreconditionILUT>();
            pc->initialize(matrix, PreconditionILUT::AdditionalData(s.ilut_drop, s.ilu_fill, s.ilu_atol, s.ilu_rtol, s.smoother_overlap));
            return detail::wrap_native_preconditioner<Vector>(pc);
          }
        if (s.trilinos_type == "IC")
          {
            auto pc = std::make_shared<PreconditionIC>();
            pc->initialize(matrix, PreconditionIC::AdditionalData(s.ilu_fill, s.ilu_atol, s.ilu_rtol, s.smoother_overlap));
            return detail::wrap_native_preconditioner<Vector>(pc);
          }
        throw std::runtime_error("Unsupported Trilinos preconditioner: " + s.trilinos_type);

    }
  };
#endif

#ifdef STOKES_HAS_PSBLAS
  template <>
  struct BackendTraits<Backend::psctoolkit>
    : detail::NativeBackendTypes<dealii::PSCToolkitWrappers::SparseMatrix,
                                 dealii::PSCToolkitWrappers::BlockSparseMatrix,
                                 dealii::PSCToolkitWrappers::Vector,
                                 dealii::PSCToolkitWrappers::BlockVector>
  {
    static std::unique_ptr<Preconditioner<Vector>>
    make_preconditioner(const Matrix &matrix, const PreconditionerSettings &settings)
    {
      using namespace detail;
      const auto comm = matrix.get_mpi_communicator();
      auto *descriptor = matrix.get_psblas_descriptor();
      require(matrix.m() == matrix.n() && descriptor == matrix.get_psblas_column_descriptor(),
              "AMG4PSBLAS needs a compressed square matrix with matching ownership");
      const auto options = native_options(settings.psc, comm);
      const auto &native = settings.psc;
      std::string ptype = native.ptype;
      std::transform(ptype.begin(), ptype.end(), ptype.begin(), [](unsigned char c) { return std::toupper(c); });
      // Preserve the native preconditioner choice and all setup phases; the
      // standard AdditionalData interface exposes only a subset of these.
      auto state = std::make_shared<PscAMG>();
      // The Stokes application uses MPI_COMM_WORLD, whose context is owned
      // by deal.II's MPI_InitFinalize and shared with its matrix wrappers.
      psc_check(amg_c_dprecinit(*dealii::InitFinalize::get_psblas_context(), state->pc, native.ptype.c_str()), "amg_c_dprecinit");
      apply_native_options(state->pc, options, "hierarchy");
      const bool split_setup = ptype == "ML" || !native.afmt.empty() || !native.chfmt.empty();
      auto *native_matrix = matrix.get_psblas_matrix();
      if (split_setup)
        psc_check(amg_c_dhierarchy_build(native_matrix, descriptor, state->pc), "amg_c_dhierarchy_build");
      apply_native_options(state->pc, options, "smoother");
      if (!native.afmt.empty() || !native.chfmt.empty())
        psc_check(amg_c_dsmoothers_build_opt(native_matrix, descriptor, state->pc,
                   native.afmt.empty() ? "CSR" : native.afmt.c_str(),
                   native.chfmt.empty() ? "HOST" : native.chfmt.c_str()), "amg_c_dsmoothers_build_opt");
      else if (ptype == "ML")
        psc_check(amg_c_dsmoothers_build(native_matrix, descriptor, state->pc), "amg_c_dsmoothers_build");
      else
        psc_check(amg_c_dprecbld(native_matrix, descriptor, state->pc), "amg_c_dprecbld");
      if (!native.chfmt.empty()) psc_check(amg_c_dallocate_wrk(state->pc, native.chfmt.c_str()), "amg_c_dallocate_wrk");
      if (native.output_details) psc_check(amg_c_ddescr(state->pc), "amg_c_ddescr");
      return std::make_unique<FunctionPreconditioner<Vector>>(
        [state, descriptor](Vector &dst, const Vector &src) {
          dst = 0.;
          psc_check(amg_c_dprecapply(state->pc, src.get_psblas_vector(), dst.get_psblas_vector(), descriptor), "amg_c_dprecapply");
        });
    }
  };
#endif
} // namespace StokesLA
