// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from smoke-dealii, Copyright (C) 2026 Fabio Durastante.
// Native rectangular PSBLAS storage follows smoke-dealii/src/algebra.cc.
#pragma once

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_operation.h>

#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX)
#  define STOKES_HAS_PETSC
#  include <deal.II/lac/petsc_sparse_matrix.h>
#  include <deal.II/lac/petsc_vector.h>
#  include <petscpc.h>
#endif
#if defined(DEAL_II_WITH_TRILINOS) && \
    ((defined(DEAL_II_TRILINOS_WITH_EPETRA) && defined(DEAL_II_TRILINOS_WITH_ML)) || \
     !DEAL_II_VERSION_GTE(9, 7, 0))
#  define STOKES_HAS_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_vector.h>
#  include <deal.II/lac/trilinos_precondition.h>
#endif
#if defined(DEAL_II_WITH_PSBLAS) && defined(DEAL_II_WITH_AMG4PSBLAS)
#  define STOKES_HAS_PSBLAS
#  include <psb_base_cbind.h>
#  include <amg_c_dprec.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
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
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using BlockVector = dealii::LinearAlgebra::distributed::BlockVector<double>;
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
        {"MAX_LEVS","integer","Maximum hierarchy levels; this pinned ML builder needs at least two. Native AS, BJAC, etc. provide one-level actions."},
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

  class Preconditioner
  {
  public:
    virtual ~Preconditioner() = default;
    virtual void vmult(Vector &dst, const Vector &src) const = 0;
  };

  class FunctionPreconditioner final : public Preconditioner
  {
  public:
    explicit FunctionPreconditioner(std::function<void(Vector &, const Vector &)> action)
      : action(std::move(action)) {}
    void vmult(Vector &dst, const Vector &src) const override { action(dst, src); }
  private:
    std::function<void(Vector &, const Vector &)> action;
  };

  namespace detail
  {
    class NativeMatrix
    {
    public:
      virtual ~NativeMatrix() = default;
      virtual void add(size_type row, size_type n, const size_type *columns, const double *values) = 0;
      virtual void compress() = 0;
      virtual void action(Vector &dst, const Vector &src, bool transpose) const = 0;
      virtual std::unique_ptr<Preconditioner> preconditioner(const PreconditionerSettings &) const = 0;
    };

    inline void copy_to_native(double *destination, const Vector &source)
    {
      for (size_type i = 0; i < source.locally_owned_size(); ++i)
        destination[i] = source.local_element(i);
    }
    inline void copy_from_native(Vector &destination, const double *source)
    {
      for (size_type i = 0; i < destination.locally_owned_size(); ++i)
        destination.local_element(i) = source[i];
    }
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

    class PetscMatrix final : public NativeMatrix
    {
    public:
      PetscMatrix(const dealii::IndexSet &rows, const dealii::IndexSet &columns,
                  const dealii::DynamicSparsityPattern &graph, MPI_Comm comm)
      {
        matrix.reinit(rows, columns, graph, comm);
        input.reinit(columns, comm);
        output.reinit(rows, comm);
      }
      void add(size_type row, size_type count, const size_type *columns, const double *values) override
      { matrix.add(row, count, columns, values, true, false); }
      void compress() override { matrix.compress(dealii::VectorOperation::add); }
      void action(Vector &dst, const Vector &src, bool transpose) const override
      {
        auto &x = transpose ? output : input;
        auto &y = transpose ? input : output;
        pack(x, src);
        if (transpose) matrix.Tvmult(y, x);
        else matrix.vmult(y, x);
        unpack(dst, y);
      }
      std::unique_ptr<Preconditioner> preconditioner(const PreconditionerSettings &settings) const override
      {
        auto state = std::make_shared<PetscPC>();
        petsc_check(PCCreate(matrix.get_mpi_communicator(), &state->pc), "PCCreate");
        petsc_check(PCSetOperators(state->pc, static_cast<Mat>(matrix), static_cast<Mat>(matrix)), "PCSetOperators");
        petsc_check(PCSetType(state->pc, settings.petsc_pc_type.c_str()), "PCSetType");
        petsc_check(PetscOptionsCreate(&state->options), "PetscOptionsCreate");
        petsc_check(PetscOptionsInsertString(state->options, settings.petsc_options.c_str()), "PetscOptionsInsertString");
        petsc_check(PetscObjectSetOptions(reinterpret_cast<PetscObject>(state->pc), state->options), "PetscObjectSetOptions");
        petsc_check(PCSetFromOptions(state->pc), "PCSetFromOptions");
        petsc_check(PCSetUp(state->pc), "PCSetUp");
        return std::make_unique<FunctionPreconditioner>(
          [this, state](Vector &dst, const Vector &src) {
            pack(input, src);
            petsc_check(PCApply(state->pc, static_cast<Vec>(input), static_cast<Vec>(output)), "PCApply");
            unpack(dst, output);
          });
      }
    private:
      static void pack(dealii::PETScWrappers::MPI::Vector &dst, const Vector &src)
      {
        PetscScalar *values = nullptr;
        petsc_check(VecGetArray(static_cast<Vec>(dst), &values), "VecGetArray");
        for (size_type i = 0; i < src.locally_owned_size(); ++i) values[i] = src.local_element(i);
        petsc_check(VecRestoreArray(static_cast<Vec>(dst), &values), "VecRestoreArray");
      }
      static void unpack(Vector &dst, const dealii::PETScWrappers::MPI::Vector &src)
      {
        const PetscScalar *values = nullptr;
        petsc_check(VecGetArrayRead(static_cast<Vec>(src), &values), "VecGetArrayRead");
        for (size_type i = 0; i < dst.locally_owned_size(); ++i) dst.local_element(i) = values[i];
        petsc_check(VecRestoreArrayRead(static_cast<Vec>(src), &values), "VecRestoreArrayRead");
      }
      dealii::PETScWrappers::MPI::SparseMatrix matrix;
      mutable dealii::PETScWrappers::MPI::Vector input, output;
    };
#endif

#ifdef STOKES_HAS_TRILINOS
    class TrilinosMatrix final : public NativeMatrix
    {
    public:
      TrilinosMatrix(const dealii::IndexSet &rows, const dealii::IndexSet &columns,
                     const dealii::DynamicSparsityPattern &graph, MPI_Comm comm)
      {
        matrix.reinit(rows, columns, graph, comm);
        input.reinit(columns, comm);
        output.reinit(rows, comm);
      }
      void add(size_type row, size_type count, const size_type *columns, const double *values) override
      { matrix.add(row, count, columns, values, true, false); }
      void compress() override { matrix.compress(dealii::VectorOperation::add); }
      void action(Vector &dst, const Vector &src, bool transpose) const override
      {
        auto &x = transpose ? output : input;
        auto &y = transpose ? input : output;
        copy_to_native(x.begin(), src);
        if (transpose) matrix.Tvmult(y, x);
        else matrix.vmult(y, x);
        copy_from_native(dst, y.begin());
      }
      std::unique_ptr<Preconditioner> preconditioner(const PreconditionerSettings &s) const override
      {
        using namespace dealii::TrilinosWrappers;
        if (s.trilinos_type == "Identity")
          return std::make_unique<FunctionPreconditioner>([](Vector &dst, const Vector &src) { dst = src; });
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
            return wrap(pc);
#else
            throw std::runtime_error("Trilinos AMG requires ML in the deal.II build");
#endif
          }
        if (s.trilinos_type == "Jacobi")
          {
            auto pc = std::make_shared<PreconditionJacobi>();
            pc->initialize(matrix, PreconditionJacobi::AdditionalData(s.relaxation, 0., s.smoother_sweeps));
            return wrap(pc);
          }
        if (s.trilinos_type == "SSOR")
          {
            auto pc = std::make_shared<PreconditionSSOR>();
            pc->initialize(matrix, PreconditionSSOR::AdditionalData(s.relaxation, 0., s.smoother_overlap, s.smoother_sweeps));
            return wrap(pc);
          }
        if (s.trilinos_type == "ILU")
          {
            auto pc = std::make_shared<PreconditionILU>();
            pc->initialize(matrix, PreconditionILU::AdditionalData(s.ilu_fill, s.ilu_atol, s.ilu_rtol, s.smoother_overlap));
            return wrap(pc);
          }
        if (s.trilinos_type == "ILUT")
          {
            auto pc = std::make_shared<PreconditionILUT>();
            pc->initialize(matrix, PreconditionILUT::AdditionalData(s.ilut_drop, s.ilu_fill, s.ilu_atol, s.ilu_rtol, s.smoother_overlap));
            return wrap(pc);
          }
        if (s.trilinos_type == "IC")
          {
            auto pc = std::make_shared<PreconditionIC>();
            pc->initialize(matrix, PreconditionIC::AdditionalData(s.ilu_fill, s.ilu_atol, s.ilu_rtol, s.smoother_overlap));
            return wrap(pc);
          }
        throw std::runtime_error("Unsupported Trilinos preconditioner: " + s.trilinos_type);
      }
    private:
      template <class PC>
      std::unique_ptr<Preconditioner> wrap(std::shared_ptr<PC> pc) const
      {
        return std::make_unique<FunctionPreconditioner>(
          [this, pc](Vector &dst, const Vector &src) {
            copy_to_native(input.begin(), src);
            output = 0.;
            pc->vmult(output, input);
            copy_from_native(dst, output.begin());
          });
      }
      dealii::TrilinosWrappers::SparseMatrix matrix;
      mutable dealii::TrilinosWrappers::MPI::Vector input, output;
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

    // Each descriptor retains the PSBLAS context, and each workspace retains its descriptor.
    class PscContext
    {
    public:
      explicit PscContext(MPI_Comm input_comm)
      {
        int comparison = MPI_UNEQUAL;
        require(MPI_Comm_compare(input_comm, MPI_COMM_WORLD, &comparison) == MPI_SUCCESS,
                "MPI_Comm_compare failed for PSBLAS context");
        if (comparison == MPI_IDENT || comparison == MPI_CONGRUENT)
          context = dealii::InitFinalize::get_psblas_context();
        else
          {
            require(MPI_Comm_dup(input_comm, &comm) == MPI_SUCCESS, "MPI_Comm_dup failed for PSBLAS context");
            const MPI_Fint original = MPI_Comm_c2f(comm);
            const psb_i_t converted = static_cast<psb_i_t>(original);
            require(static_cast<MPI_Fint>(converted) == original, "MPI communicator exceeds the PSBLAS integer width");
            psb_i_t rank = -1, size = -1;
            psb_c_info(*dealii::InitFinalize::get_psblas_context(), &rank, &size);
            psb_c_init_from_fint(&owned, converted);
            require(owned.ctxt != nullptr, "Cannot initialize PSBLAS subcommunicator");
            context = &owned;
          }
        require(context != nullptr, "Initialize PSBLAS using deal.II MPI_InitFinalize before creating matrices");
      }
      ~PscContext()
      {
        if (owned.ctxt) psb_c_exit_ctxt(owned);
        if (comm != MPI_COMM_NULL) (void)MPI_Comm_free(&comm);
      }
      psb_c_ctxt *get() const { return context; }
    private:
      MPI_Comm comm = MPI_COMM_NULL;
      psb_c_ctxt owned{};
      psb_c_ctxt *context = nullptr;
    };

    struct PscDescriptor
    {
      PscDescriptor(std::shared_ptr<PscContext> context, size_type local_size)
        : context(std::move(context))
      { psc_check(psb_c_cdall_nl(static_cast<psb_i_t>(local_size), *this->context->get(), &value), "psb_c_cdall_nl"); }
      ~PscDescriptor() { if (value.descriptor) (void)psb_c_cdfree(&value); }
      std::shared_ptr<PscContext> context;
      psb_c_descriptor value{};
    };

    class PscVector
    {
    public:
      void initialize(std::shared_ptr<PscDescriptor> descriptor)
      {
        this->descriptor = std::move(descriptor);
        psc_check(psb_c_dgeall_remote_options(&value, &this->descriptor->value,
                                            PSB_MATBLD_REMOTE, PSB_DUPL_DEF), "psb_c_dgeall_remote_options");
        allocated = true;
        psc_check(psb_c_dgeasb_options(&value, &this->descriptor->value, PSB_DUPL_DEF), "psb_c_dgeasb_options");
        psc_check(psb_c_dvect_zero(&value), "psb_c_dvect_zero");
      }
      ~PscVector() { if (allocated) (void)psb_c_dgefree(&value, &descriptor->value); }
      psb_c_dvector *get() const { return &value; }
      double *data() const { return psb_c_dvect_f_get_pnt(&value); }
    private:
      std::shared_ptr<PscDescriptor> descriptor;
      mutable psb_c_dvector value{};
      bool allocated = false;
    };

    struct PscAMG
    {
      explicit PscAMG(std::shared_ptr<PscDescriptor> descriptor) : descriptor(std::move(descriptor))
      {
        input.initialize(this->descriptor);
        output.initialize(this->descriptor);
        pc = amg_c_dprec_new();
        require(pc != nullptr, "AMG4PSBLAS preconditioner allocation failed");
      }
      ~PscAMG() { if (pc) (void)amg_c_dprec_delete(pc); }
      std::shared_ptr<PscDescriptor> descriptor;
      PscVector input, output;
      amg_c_dprec *pc = nullptr;
    };

    struct Contribution
    {
      size_type row = 0, column = 0;
      double value = 0.;
      template <class Archive> void serialize(Archive &archive, const unsigned int)
      { archive & row & column & value; }
    };
    struct PscStorage
    {
      ~PscStorage() { if (matrix.dspmat) (void)psb_c_dspfree(&matrix, &rows->value); }
      std::shared_ptr<PscDescriptor> rows, columns;
      psb_c_dspmat matrix{};
    };

    class PscMatrix final : public NativeMatrix
    {
    public:
      PscMatrix(const dealii::IndexSet &rows, const dealii::IndexSet &columns,
                const dealii::DynamicSparsityPattern &, MPI_Comm comm)
        : owned(rows), owned_columns(columns), comm(comm)
      {
        const bool fits = std::max(rows.size(), columns.size()) <=
                            static_cast<size_type>(std::numeric_limits<psb_l_t>::max()) &&
                          std::max(rows.n_elements(), columns.n_elements()) <=
                            static_cast<size_type>(std::numeric_limits<psb_i_t>::max());
        require(dealii::Utilities::MPI::min(static_cast<unsigned int>(fits), comm) != 0,
                "Matrix dimensions exceed this PSBLAS build's integer widths");
        auto context = std::make_shared<PscContext>(comm);
        storage.rows = std::make_shared<PscDescriptor>(context, rows.n_elements());
        storage.columns = std::make_shared<PscDescriptor>(context, columns.n_elements());
        psc_check(psb_c_dspall_2desc(&storage.matrix, &storage.rows->value, &storage.columns->value), "psb_c_dspall_2desc");
        const auto counts = dealii::Utilities::MPI::all_gather(comm, rows.n_elements());
        size_type end = 0;
        for (const auto count : counts) { end += count; owner_ends.push_back(end); }
        has_empty_rows = std::find(counts.begin(), counts.end(), 0) != counts.end();
      }

      void add(size_type row, size_type n, const size_type *columns, const double *values) override
      {
        require(!assembled, "PSBLAS matrix is already compressed; reinit before assembling another problem");
        require(n <= static_cast<size_type>(std::numeric_limits<psb_i_t>::max()), "PSBLAS insertion packet exceeds psb_i_t");
        if (!owned.is_element(row))
          {
            const auto owner = std::upper_bound(owner_ends.begin(), owner_ends.end(), row);
            require(owner != owner_ends.end(), "PSBLAS assembly row has no owner");
            auto &packet = remote[static_cast<unsigned int>(owner - owner_ends.begin())];
            for (size_type i = 0; i < n; ++i)
              if (values[i] != 0.) packet.push_back({row, columns[i], values[i]});
          }
        else
          {
            std::vector<psb_l_t> row_indices(n, static_cast<psb_l_t>(row)), column_indices(n);
            for (size_type i = 0; i < n; ++i) column_indices[i] = static_cast<psb_l_t>(columns[i]);
            if (n)
              psc_check(psb_c_dspins_2desc(static_cast<psb_i_t>(n), row_indices.data(), column_indices.data(), values,
                                           &storage.matrix, &storage.rows->value, &storage.columns->value), "psb_c_dspins_2desc");
          }
      }

      void compress() override
      {
        if (assembled) return;
        // The native two-descriptor insertion accepts locally owned rows.
        // Route all off-process finite-element contributions before assembly.
        const auto received = dealii::Utilities::MPI::some_to_some(comm, remote);
        remote.clear();
        for (const auto &[sender, packet] : received)
          {
            (void)sender;
            require(packet.size() <= static_cast<std::size_t>(std::numeric_limits<psb_i_t>::max()),
                    "PSBLAS received assembly packet exceeds psb_i_t");
            std::vector<psb_l_t> rows, columns;
            std::vector<double> values;
            rows.reserve(packet.size()); columns.reserve(packet.size()); values.reserve(packet.size());
            for (const auto &entry : packet)
              {
                require(owned.is_element(entry.row), "PSBLAS assembly packet was sent to the wrong row owner");
                rows.push_back(static_cast<psb_l_t>(entry.row));
                columns.push_back(static_cast<psb_l_t>(entry.column));
                values.push_back(entry.value);
              }
            if (!packet.empty())
              psc_check(psb_c_dspins_2desc(static_cast<psb_i_t>(packet.size()), rows.data(), columns.data(), values.data(),
                                          &storage.matrix, &storage.rows->value, &storage.columns->value), "psb_c_dspins_2desc(remote)");
          }
        psc_check(psb_c_cdasb(&storage.rows->value), "psb_c_cdasb(rows)");
        psc_check(psb_c_cdasb(&storage.columns->value), "psb_c_cdasb(columns)");
        psc_check(psb_c_dspasb_2desc(&storage.matrix, &storage.rows->value, &storage.columns->value), "psb_c_dspasb_2desc");
        require(psb_c_cd_get_global_rows(&storage.rows->value) == static_cast<psb_l_t>(owned.size()) &&
                psb_c_cd_get_global_rows(&storage.columns->value) == static_cast<psb_l_t>(owned_columns.size()),
                "Native PSBLAS descriptors do not match the finite-element fields");
        input.initialize(storage.columns);
        output.initialize(storage.rows);
        assembled = true;
      }

      void action(Vector &dst, const Vector &src, bool transpose) const override
      {
        require(assembled, "Compress the PSBLAS matrix before multiplication");
        auto &x = transpose ? output : input;
        auto &y = transpose ? input : output;
        copy_to_native(x.data(), src);
        psc_check(psb_c_dvect_zero(y.get()), "psb_c_dvect_zero");
        psc_check(psb_c_dspmm_2desc(1., const_cast<psb_c_dspmat *>(&storage.matrix), x.get(), 0., y.get(),
                                   &storage.rows->value, &storage.columns->value, transpose ? "T" : "N"), "psb_c_dspmm_2desc");
        copy_from_native(dst, y.data());
      }

      std::unique_ptr<Preconditioner> preconditioner(const PreconditionerSettings &settings) const override
      {
        require(assembled && owned == owned_columns, "AMG4PSBLAS needs a compressed square matrix with matching ownership");
        if (has_empty_rows) return active_preconditioner(settings);
        const auto options = native_options(settings.psc, comm);
        const auto &native = settings.psc;
        std::string ptype = native.ptype;
        std::transform(ptype.begin(), ptype.end(), ptype.begin(), [](unsigned char c) { return std::toupper(c); });
        // Do not disguise the pinned native ML/MAX_LEVS=1 limitation by silently selecting another preconditioner.
        if (ptype == "ML")
          {
            psb_i_t maximum_levels = 0;
            for (const auto &option : options)
              if (option.name == "MAX_LEVS" && option.type == "integer")
                maximum_levels = native_integer(option.value);
            require(maximum_levels == 0 || maximum_levels >= 2,
              "This pinned AMG4PSBLAS ML builder needs MAX_LEVS >= 2; choose the native ptype AS, BJAC or JACOBI for one-level experiments");
          }
        auto state = std::make_shared<PscAMG>(storage.columns);
        psc_check(amg_c_dprecinit(*storage.columns->context->get(), state->pc, native.ptype.c_str()), "amg_c_dprecinit");
        apply_native_options(state->pc, options, "hierarchy");
        const bool split_setup = ptype == "ML" || !native.afmt.empty() || !native.chfmt.empty();
        auto *matrix = const_cast<psb_c_dspmat *>(&storage.matrix);
        if (split_setup)
          psc_check(amg_c_dhierarchy_build(matrix, &storage.columns->value, state->pc), "amg_c_dhierarchy_build");
        apply_native_options(state->pc, options, "smoother");
        if (!native.afmt.empty() || !native.chfmt.empty())
          psc_check(amg_c_dsmoothers_build_opt(matrix, &storage.columns->value, state->pc,
                     native.afmt.empty() ? "CSR" : native.afmt.c_str(),
                     native.chfmt.empty() ? "HOST" : native.chfmt.c_str()), "amg_c_dsmoothers_build_opt");
        else if (ptype == "ML")
          psc_check(amg_c_dsmoothers_build(matrix, &storage.columns->value, state->pc), "amg_c_dsmoothers_build");
        else
          psc_check(amg_c_dprecbld(matrix, &storage.columns->value, state->pc), "amg_c_dprecbld");
        if (!native.chfmt.empty()) psc_check(amg_c_dallocate_wrk(state->pc, native.chfmt.c_str()), "amg_c_dallocate_wrk");
        if (native.output_details) psc_check(amg_c_ddescr(state->pc), "amg_c_ddescr");
        return std::make_unique<FunctionPreconditioner>(
          [state](Vector &dst, const Vector &src) {
            copy_to_native(state->input.data(), src);
            psc_check(psb_c_dvect_zero(state->output.get()), "psb_c_dvect_zero(AMG)");
            psc_check(amg_c_dprecapply(state->pc, state->input.get(), state->output.get(), &state->descriptor->value), "amg_c_dprecapply");
            copy_from_native(dst, state->output.data());
          });
      }

    private:
      std::unique_ptr<Preconditioner> active_preconditioner(const PreconditionerSettings &settings) const
      {
        // AMG kernels need owned rows. Retain the same global matrix and
        // eliminate only ranks with zero rows from the auxiliary communicator.
        struct Active
        {
          ~Active()
          {
            pc.reset(); matrix.reset();
            if (comm != MPI_COMM_NULL) (void)MPI_Comm_free(&comm);
          }
          MPI_Comm comm = MPI_COMM_NULL;
          std::unique_ptr<PscMatrix> matrix;
          std::unique_ptr<Preconditioner> pc;
        };
        auto state = std::make_shared<Active>();
        const int rank = static_cast<int>(dealii::Utilities::MPI::this_mpi_process(comm));
        require(MPI_Comm_split(comm, owned.n_elements() ? 0 : MPI_UNDEFINED, rank, &state->comm) == MPI_SUCCESS,
                "MPI_Comm_split failed for AMG4PSBLAS active rows");
        if (state->comm != MPI_COMM_NULL)
          {
            psb_i_t count = 0;
            psb_l_t dummy = 0;
            double value = 0.;
            auto *matrix = const_cast<psb_c_dspmat *>(&storage.matrix);
            psc_check(psb_c_dspget_coo(matrix, &storage.rows->value, &storage.columns->value,
                                      0, &count, &dummy, &dummy, &value), "psb_c_dspget_coo(size)");
            require(count >= 0, "Invalid native sparse export size");
            const auto capacity = std::max<psb_i_t>(1, count);
            std::vector<psb_l_t> rows(capacity), columns(capacity);
            std::vector<double> values(capacity);
            psc_check(psb_c_dspget_coo(matrix, &storage.rows->value, &storage.columns->value,
                                      count, &count, rows.data(), columns.data(), values.data()), "psb_c_dspget_coo");
            dealii::DynamicSparsityPattern graph(owned);
            for (psb_i_t i = 0; i < count; ++i) graph.add(rows[i], columns[i]);
            state->matrix = std::make_unique<PscMatrix>(owned, owned, graph, state->comm);
            for (psb_i_t i = 0; i < count; ++i)
              {
                const size_type column = static_cast<size_type>(columns[i]);
                state->matrix->add(static_cast<size_type>(rows[i]), 1, &column, &values[i]);
              }
            state->matrix->compress();
            state->pc = state->matrix->preconditioner(settings);
          }
        return std::make_unique<FunctionPreconditioner>([state](Vector &dst, const Vector &src) {
          if (state->pc) state->pc->vmult(dst, src);
          else dst = 0.;
        });
      }

      dealii::IndexSet owned, owned_columns;
      MPI_Comm comm;
      std::vector<size_type> owner_ends;
      std::map<unsigned int, std::vector<Contribution>> remote;
      PscStorage storage;
      mutable PscVector input, output;
      bool assembled = false, has_empty_rows = false;
    };
#endif
  } // namespace detail

  /** One actual native rectangular block. All vectors keep compact field numbering. */
  class Matrix
  {
  public:
    using value_type = double;
    using size_type = StokesLA::size_type;
    explicit Matrix(Backend backend) : backend_(backend) {}
    virtual ~Matrix() = default;
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    void reinit(const dealii::IndexSet &rows, const dealii::IndexSet &columns,
                const dealii::DynamicSparsityPattern &graph, MPI_Comm comm)
    {
      const auto check_partition = [&](const dealii::IndexSet &partition) {
        const auto sizes = dealii::Utilities::MPI::all_gather(comm, partition.n_elements());
        const auto rank = dealii::Utilities::MPI::this_mpi_process(comm);
        size_type beginning = 0, total = 0;
        for (unsigned int r = 0; r < sizes.size(); ++r)
          { if (r < rank) beginning += sizes[r]; total += sizes[r]; }
        const bool valid = partition.is_contiguous() && total == partition.size() &&
                          (partition.n_elements() == 0 || *partition.begin() == beginning);
        require(dealii::Utilities::MPI::min(static_cast<unsigned int>(valid), comm) != 0,
                "Field ownership must be contiguous within each rank and ordered by MPI rank");
      };
      check_partition(rows); check_partition(columns);
      row_partition = rows; column_partition = columns;
      switch (backend_)
        {
#ifdef STOKES_HAS_PETSC
          case Backend::petsc: native = std::make_unique<detail::PetscMatrix>(rows, columns, graph, comm); break;
#endif
#ifdef STOKES_HAS_TRILINOS
          case Backend::trilinos: native = std::make_unique<detail::TrilinosMatrix>(rows, columns, graph, comm); break;
#endif
#ifdef STOKES_HAS_PSBLAS
          case Backend::psctoolkit: native = std::make_unique<detail::PscMatrix>(rows, columns, graph, comm); break;
#endif
          default: throw std::runtime_error("Requested backend was not compiled into this executable");
        }
    }

    size_type m() const { return row_partition.size(); }
    size_type n() const { return column_partition.size(); }
    Backend backend() const { return backend_; }
    const dealii::IndexSet &locally_owned_range_indices() const { return row_partition; }
    const dealii::IndexSet &locally_owned_domain_indices() const { return column_partition; }

    void add(size_type row, size_type column, double value)
    { add(row, 1, &column, &value); }
    void add(size_type row, size_type count, const size_type *columns, const double *values,
             bool = true, bool = false)
    { require(native != nullptr, "Matrix has not been initialized"); native->add(row, count, columns, values); }
    void compress(dealii::VectorOperation::values operation = dealii::VectorOperation::add)
    { require(operation == dealii::VectorOperation::add, "Matrix assembly uses additive compression"); native->compress(); }
    void vmult(Vector &dst, const Vector &src) const { native->action(dst, src, false); }
    void Tvmult(Vector &dst, const Vector &src) const { native->action(dst, src, true); }
    std::unique_ptr<Preconditioner> make_preconditioner(const PreconditionerSettings &settings) const
    {
      require(row_partition == column_partition, "A block preconditioner needs matching square partitions");
      return native->preconditioner(settings);
    }
  private:
    Backend backend_;
    dealii::IndexSet row_partition, column_partition;
    std::unique_ptr<detail::NativeMatrix> native;
  };

  /** Two-field distributed block matrix assembled in mixed finite-element numbering. */
  class BlockMatrix
  {
  public:
    using value_type = double;
    using size_type = StokesLA::size_type;
    explicit BlockMatrix(Backend backend) : backend_(backend) {}
    virtual ~BlockMatrix() = default;

    void reinit(const std::vector<dealii::IndexSet> &partition,
                const dealii::BlockDynamicSparsityPattern &graph, MPI_Comm comm)
    {
      require(partition.size() == 2, "This Stokes example uses velocity and pressure blocks");
      offsets = {{0, partition[0].size(), partition[0].size() + partition[1].size()}};
      for (unsigned int i = 0; i < 2; ++i)
        {
          temporary[i].reinit(partition[i], comm);
          for (unsigned int j = 0; j < 2; ++j)
            {
              matrices[2 * i + j] = std::make_unique<Matrix>(backend_);
              matrices[2 * i + j]->reinit(partition[i], partition[j], graph.block(i, j), comm);
            }
        }
    }
    size_type m() const { return offsets[2]; }
    size_type n() const { return offsets[2]; }
    unsigned int n_block_rows() const { return 2; }
    unsigned int n_block_cols() const { return 2; }
    Matrix &block(unsigned int i, unsigned int j) { return *matrices.at(2 * i + j); }
    const Matrix &block(unsigned int i, unsigned int j) const { return *matrices.at(2 * i + j); }

    void add(size_type row, size_type column, double value)
    {
      const unsigned int i = row < offsets[1] ? 0 : 1, j = column < offsets[1] ? 0 : 1;
      block(i, j).add(row - offsets[i], column - offsets[j], value);
    }
    void add(size_type row, size_type count, const size_type *columns, const double *values,
             bool elide_zero_values = true, bool = false)
    {
      const unsigned int i = row < offsets[1] ? 0 : 1;
      std::array<std::vector<size_type>, 2> indices;
      std::array<std::vector<double>, 2> entries;
      for (size_type k = 0; k < count; ++k)
        if (!elide_zero_values || values[k] != 0.)
          {
            const unsigned int j = columns[k] < offsets[1] ? 0 : 1;
            indices[j].push_back(columns[k] - offsets[j]);
            entries[j].push_back(values[k]);
          }
      for (unsigned int j = 0; j < 2; ++j)
        if (!indices[j].empty())
          block(i, j).add(row - offsets[i], indices[j].size(), indices[j].data(), entries[j].data());
    }
    void compress(dealii::VectorOperation::values operation = dealii::VectorOperation::add)
    { for (auto &matrix : matrices) matrix->compress(operation); }
    void vmult(BlockVector &dst, const BlockVector &src) const
    {
      for (unsigned int i = 0; i < 2; ++i)
        {
          dst.block(i) = 0.;
          for (unsigned int j = 0; j < 2; ++j)
            {
              block(i, j).vmult(temporary[i], src.block(j));
              dst.block(i) += temporary[i];
            }
        }
    }
  private:
    Backend backend_;
    std::array<size_type, 3> offsets{};
    std::array<std::unique_ptr<Matrix>, 4> matrices;
    mutable std::array<Vector, 2> temporary;
  };
} // namespace StokesLA
