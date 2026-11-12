// SPDX-License-Identifier: GPL-3.0-or-later
// The backend adapter is derived from smoke-dealii; see LICENSE.stokes.
// A deliberately small, MPI-distributed laboratory for Stokes block solvers.

#include "stokes_backends.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.templates.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>

namespace Stokes
{
  using namespace dealii;
  using Vector = StokesLA::Vector;
  using BlockVector = StokesLA::BlockVector;

  struct BlockSettings
  {
    std::string action = "solve";
    std::string solver = "fgmres";
    unsigned int max_iterations = 200;
    unsigned int restart = 30;
    double relative_tolerance = 1.e-2;
    double absolute_tolerance = 1.e-14;
    StokesLA::PreconditionerSettings preconditioner;

    explicit BlockSettings(const bool pressure = false)
      : preconditioner(pressure)
    {}

    void declare_parameters(ParameterHandler &prm)
    {
      prm.add_parameter("Action", action,
                        "apply: one native preconditioner application; solve: an auxiliary Krylov solve.",
                        Patterns::Selection("apply|solve"));
      prm.add_parameter("Solver", solver,
                        "Auxiliary solver, used only by Action=solve. CG requires an SPD preconditioner.",
                        Patterns::Selection("cg|gmres|fgmres"));
      prm.add_parameter("Maximum iterations", max_iterations, "Auxiliary iteration limit.", Patterns::Integer(1));
      prm.add_parameter("Restart length", restart, "Auxiliary GMRES/FGMRES basis length.", Patterns::Integer(2));
      prm.add_parameter("Relative tolerance", relative_tolerance,
                        "Auxiliary target is max(absolute tolerance, relative tolerance * norm(rhs)).",
                        Patterns::Double(0., 1.));
      prm.add_parameter("Absolute tolerance", absolute_tolerance, "Auxiliary absolute residual tolerance.", Patterns::Double(0.));
      preconditioner.declare_parameters(prm);
    }
  };

  struct Settings
  {
    std::string backend = StokesLA::backend_name(StokesLA::default_backend());
    std::string problem = "poiseuille";
    std::string lid = "regularized";
    double viscosity = 1.;
    double step_inlet_peak = 1.;
    unsigned int pressure_degree = 1;
    unsigned int initial_refinements = 2;
    unsigned int cycles = 1;
    std::string block_preconditioner = "upper triangular";
    unsigned int max_iterations = 500;
    unsigned int restart = 50;
    double relative_tolerance = 1.e-9;
    double absolute_tolerance = 1.e-12;
    BlockSettings velocity{false};
    BlockSettings pressure{true};
    std::string output_directory = "output";
    bool write_visualization = true;
    bool write_history = true;
    bool solver_trace = false;
    double maximum_velocity_error = -1.;
    double maximum_pressure_error = -1.;
    double maximum_divergence = -1.;

    bool closed_boundary() const
    {
      return problem == "cavity" || problem == "colliding";
    }

    bool has_exact_solution() const
    {
      return problem == "poiseuille" || problem == "colliding";
    }

    void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Problem");
      prm.add_parameter("Test case", problem,
                        "Examples 5.1.1--5.1.4 in the supplied Stokes excerpt, all with zero body force.",
                        Patterns::Selection("poiseuille|backward_step|cavity|colliding"));
      prm.add_parameter("Cavity lid", lid,
                        "leaky: unit top-corner values; watertight: zero corner values; regularized: 1-x^4.",
                        Patterns::Selection("leaky|watertight|regularized"));
      prm.add_parameter("Viscosity", viscosity, "Positive constant nu in -nu*Laplacian(u)+grad(p)=0.", Patterns::Double(0.));
      prm.add_parameter("Step inlet peak velocity", step_inlet_peak,
                        "The step inlet is 4*peak*y*(1-y). The excerpt specifies its shape but not amplitude.",
                        Patterns::Double(0.));
      prm.leave_subsection();

      prm.enter_subsection("Discretization");
      prm.add_parameter("Pressure degree", pressure_degree,
                        "Only continuous Taylor-Hood [Q_(k+1)]^2/Q_k, k>=1. Velocity degree is derived.",
                        Patterns::Integer(1));
      prm.add_parameter("Initial global refinements", initial_refinements,
                        "Uniform refinements of a 2x2 square, or an eleven-cell backward-step mesh.", Patterns::Integer(0));
      prm.add_parameter("Refinement cycles", cycles,
                        "Solve on successively uniformly refined meshes; each cycle starts from zero.", Patterns::Integer(1));
      prm.leave_subsection();

      prm.enter_subsection("Linear algebra");
      prm.add_parameter("Backend", backend,
                        "Runtime backend. The chosen backend must be enabled in the linked deal.II build.",
                        Patterns::Selection("PETSc|Trilinos|PSBLAS"));
      prm.add_parameter("Block preconditioner", block_preconditioner,
                        "diagonal uses diag(A,Q); triangular forms use -Q with Q=M_p/nu.",
                        Patterns::Selection("diagonal|upper triangular|lower triangular"));
      prm.enter_subsection("FGMRES");
      prm.add_parameter("Maximum iterations", max_iterations, "Outer flexible GMRES iteration limit.", Patterns::Integer(1));
      prm.add_parameter("Restart length", restart, "Outer flexible GMRES basis length.", Patterns::Integer(2));
      prm.add_parameter("Relative tolerance", relative_tolerance, "Outer relative residual tolerance.", Patterns::Double(0., 1.));
      prm.add_parameter("Absolute tolerance", absolute_tolerance, "Outer absolute residual tolerance.", Patterns::Double(0.));
      prm.leave_subsection();
      prm.enter_subsection("Velocity block");
      velocity.declare_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("Pressure block");
      pressure.declare_parameters(prm);
      prm.leave_subsection();
      prm.leave_subsection();

      prm.enter_subsection("Output");
      prm.add_parameter("Directory", output_directory, "Relative paths are relative to the process working directory.");
      prm.add_parameter("Write visualization", write_visualization, "Write parallel VTU/PVTU files for velocity and pressure.");
      prm.add_parameter("Write residual history", write_history, "Write the outer FGMRES residual estimates to CSV.");
      prm.add_parameter("Solver trace", solver_trace, "Print deal.II solver progress on rank zero.");
      prm.leave_subsection();

      prm.enter_subsection("Verification");
      prm.add_parameter("Maximum velocity L2 error", maximum_velocity_error,
                        "Negative disables this check; available for poiseuille and colliding.");
      prm.add_parameter("Maximum pressure L2 error", maximum_pressure_error,
                        "Negative disables this check; colliding pressure errors are modulo constants.");
      prm.add_parameter("Maximum divergence L2 norm", maximum_divergence,
                        "Negative disables this check. Taylor-Hood is weakly, not pointwise, divergence-free.");
      prm.leave_subsection();
    }

    void validate() const
    {
      AssertThrow(std::isfinite(viscosity) && viscosity > 0., ExcMessage("Viscosity must be positive and finite."));
      AssertThrow(std::isfinite(step_inlet_peak) && step_inlet_peak > 0., ExcMessage("Step inlet peak velocity must be positive and finite."));
      for (const auto *block : {&velocity, &pressure})
        AssertThrow(block->absolute_tolerance > 0. || block->relative_tolerance > 0.,
                    ExcMessage("Each auxiliary solver needs a positive stopping tolerance."));
      AssertThrow(absolute_tolerance > 0. || relative_tolerance > 0.,
                  ExcMessage("FGMRES needs a positive stopping tolerance."));
      AssertThrow(!output_directory.empty(), ExcMessage("Output/Directory must not be empty."));
      AssertThrow(has_exact_solution() || (maximum_velocity_error < 0. && maximum_pressure_error < 0.),
                  ExcMessage("Exact-solution checks require poiseuille or colliding."));
      (void)StokesLA::parse_backend(backend);
      velocity.preconditioner.validate();
      pressure.preconditioner.validate();
    }
  };

  // One boundary function makes both top corners independent of face ordering.
  class BoundaryValues : public Function<2>
  {
  public:
    explicit BoundaryValues(const Settings &settings) : Function<2>(3), p(settings) {}

    double value(const Point<2> &x, const unsigned int component = 0) const override
    {
      if (component == 2) return 0.;
      if (p.problem == "colliding")
        return component == 0 ? 20.*x[0]*std::pow(x[1], 3) : 5.*(std::pow(x[0], 4)-std::pow(x[1], 4));
      if (component == 1) return 0.;
      if (p.problem == "poiseuille") return 1.-x[1]*x[1];
      if (p.problem == "backward_step")
        return std::abs(x[0]+1.) < 1.e-12 ? 4.*p.step_inlet_peak*x[1]*(1.-x[1]) : 0.;
      if (std::abs(x[1]-1.) > 1.e-12) return 0.;
      if (p.lid == "regularized") return 1.-std::pow(x[0], 4);
      if (p.lid == "watertight" && std::abs(std::abs(x[0])-1.) < 1.e-12) return 0.;
      return 1.;
    }

    void vector_value(const Point<2> &point, dealii::Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < 3; ++c) values[c] = value(point, c);
    }

  private:
    const Settings &p;
  };

  class ExactSolution : public Function<2>
  {
  public:
    explicit ExactSolution(const Settings &settings) : Function<2>(3), p(settings) {}

    double value(const Point<2> &x, const unsigned int component = 0) const override
    {
      if (p.problem == "poiseuille")
        return component == 0 ? 1.-x[1]*x[1] : (component == 1 ? 0. : 2.*p.viscosity*(1.-x[0]));
      if (component == 0) return 20.*x[0]*std::pow(x[1], 3);
      if (component == 1) return 5.*(std::pow(x[0], 4)-std::pow(x[1], 4));
      return p.viscosity*(60.*x[0]*x[0]*x[1]-20.*std::pow(x[1], 3));
    }

    Tensor<1, 2> gradient(const Point<2> &x, const unsigned int component = 0) const override
    {
      Tensor<1, 2> g;
      if (p.problem == "poiseuille")
        {
          if (component == 0) g[1] = -2.*x[1];
          if (component == 2) g[0] = -2.*p.viscosity;
        }
      else if (component == 0)
        { g[0] = 20.*std::pow(x[1], 3); g[1] = 60.*x[0]*x[1]*x[1]; }
      else if (component == 1)
        { g[0] = 20.*std::pow(x[0], 3); g[1] = -20.*std::pow(x[1], 3); }
      else
        { g[0] = 120.*p.viscosity*x[0]*x[1]; g[1] = 60.*p.viscosity*(x[0]*x[0]-x[1]*x[1]); }
      return g;
    }

  private:
    const Settings &p;
  };

  class BlockInverse
  {
  public:
    BlockInverse(const StokesLA::Matrix &matrix, const BlockSettings &settings,
                 const std::string &label)
      : matrix(matrix), settings(settings), label(label),
        preconditioner(matrix.make_preconditioner(settings.preconditioner))
    {}

    void vmult(Vector &dst, const Vector &src) const
    {
      ++applications;
      dst = 0.;
      const double norm = src.l2_norm();
      if (norm == 0.) return;
      if (settings.action == "apply")
        {
          preconditioner->vmult(dst, src);
          return;
        }
      const double tolerance = std::max(settings.absolute_tolerance, settings.relative_tolerance*norm);
      SolverControl control(settings.max_iterations, tolerance, false, false);
      try
        {
          if (settings.solver == "cg")
            {
              SolverCG<Vector> solver(control);
              solver.solve(matrix, dst, src, *preconditioner);
            }
          else if (settings.solver == "gmres")
            {
              SolverGMRES<Vector> solver(control, typename SolverGMRES<Vector>::AdditionalData(settings.restart));
              solver.solve(matrix, dst, src, *preconditioner);
            }
          else
            {
              SolverFGMRES<Vector> solver(control, typename SolverFGMRES<Vector>::AdditionalData(settings.restart));
              solver.solve(matrix, dst, src, *preconditioner);
            }
        }
      catch (const SolverControl::NoConvergence &)
        {
          AssertThrow(false, ExcMessage(label + " auxiliary solve failed after " +
                                        std::to_string(control.last_step()) + " iterations; residual " +
                                        std::to_string(control.last_value()) + "."));
        }
      iterations += control.last_step();
      largest_iteration_count = std::max(largest_iteration_count, control.last_step());
    }

    mutable unsigned long long applications = 0;
    mutable unsigned long long iterations = 0;
    mutable unsigned int largest_iteration_count = 0;

  private:
    const StokesLA::Matrix &matrix;
    const BlockSettings &settings;
    const std::string label;
    std::unique_ptr<StokesLA::Preconditioner> preconditioner;
  };

  class BlockPreconditioner
  {
  public:
    BlockPreconditioner(const StokesLA::BlockMatrix &matrix,
                        const BlockInverse &velocity, const BlockInverse &pressure,
                        const std::string &form, const BlockVector &layout)
      : matrix(matrix), velocity(velocity), pressure(pressure), form(form)
    {
      work.reinit(layout);
    }

    void vmult(BlockVector &dst, const BlockVector &src) const
    {
      // B_ij are the assembled signed blocks. Q=M_p/nu approximates
      // +B*A^{-1}*B^T, so the triangular pressure diagonal is -Q.
      if (form == "diagonal")
        {
          velocity.vmult(dst.block(0), src.block(0));
          pressure.vmult(dst.block(1), src.block(1));
        }
      else if (form == "upper triangular")
        {
          pressure.vmult(dst.block(1), src.block(1));
          dst.block(1) *= -1.;
          matrix.block(0, 1).vmult(work.block(0), dst.block(1));
          work.block(0).sadd(-1., 1., src.block(0));
          velocity.vmult(dst.block(0), work.block(0));
        }
      else
        {
          velocity.vmult(dst.block(0), src.block(0));
          matrix.block(1, 0).vmult(work.block(1), dst.block(0));
          work.block(1).sadd(-1., 1., src.block(1));
          pressure.vmult(dst.block(1), work.block(1));
          dst.block(1) *= -1.;
        }
    }

  private:
    const StokesLA::BlockMatrix &matrix;
    const BlockInverse &velocity, &pressure;
    const std::string form;
    mutable BlockVector work;
  };

  class Problem
  {
  public:
    explicit Problem(const Settings &settings)
      : p(settings), comm(MPI_COMM_WORLD),
        pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
        triangulation(comm),
        fe(FE_Q<2>(p.pressure_degree+1), 2, FE_Q<2>(p.pressure_degree), 1),
        dofs(triangulation),
        system_matrix(StokesLA::parse_backend(p.backend)),
        pressure_mass(StokesLA::parse_backend(p.backend))
    {}

    void run(const ParameterHandler &parameters);

  private:
    void make_grid();
    void setup();
    void assemble();
    void solve(unsigned int cycle);
    void diagnostics(unsigned int cycle);
    void output(unsigned int cycle) const;
    void rank_zero_io(const std::function<void()> &operation) const;

    const Settings &p;
    const MPI_Comm comm;
    ConditionalOStream pcout;
    parallel::distributed::Triangulation<2> triangulation;
    FESystem<2> fe;
    DoFHandler<2> dofs;
    AffineConstraints<double> constraints, pressure_constraints;
    IndexSet owned, relevant;
    std::vector<IndexSet> owned_blocks, relevant_blocks;
    types::global_dof_index n_velocity = 0;
    StokesLA::BlockMatrix system_matrix;
    StokesLA::Matrix pressure_mass;
    BlockVector solution, ghosted_solution, rhs, assembly_rhs;
    const FEValuesExtractors::Vector velocity{0};
    const FEValuesExtractors::Scalar pressure{2};
    double residual_norm = 0., rhs_norm = 0., setup_seconds = 0., solve_seconds = 0.;
    unsigned int outer_iterations = 0;
    unsigned long long velocity_iterations = 0, pressure_iterations = 0;
    std::ofstream summary;
  };

  void Problem::rank_zero_io(const std::function<void()> &operation) const
  {
    std::string error;
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      try { operation(); }
      catch (const std::exception &exception) { error = exception.what(); }
    error = Utilities::MPI::broadcast(comm, error, 0);
    AssertThrow(error.empty(), ExcMessage(error));
  }

  void Problem::make_grid()
  {
    if (p.problem == "backward_step")
      {
        Triangulation<2> rectangle;
        GridGenerator::subdivided_hyper_rectangle(rectangle, std::vector<unsigned int>{6, 2},
                                                  Point<2>(-1., -1.), Point<2>(5., 1.));
        std::set<Triangulation<2>::active_cell_iterator> removed;
        for (const auto &cell : rectangle.active_cell_iterators())
          if (cell->center()[0] < 0. && cell->center()[1] < 0.) removed.insert(cell);
        GridGenerator::create_triangulation_with_removed_cells(rectangle, removed, triangulation);
      }
    else
      GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{2, 2},
                                                Point<2>(-1., -1.), Point<2>(1., 1.));

    // Mark the coarse mesh before refinement, so children inherit the IDs.
    for (const auto &cell : triangulation.active_cell_iterators())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary())
            {
              auto boundary = cell->face(face);
              const auto center = boundary->center();
              boundary->set_boundary_id(0);
              if (!p.closed_boundary())
                {
                  if (std::abs(center[0]+1.) < 1.e-12) boundary->set_boundary_id(1);
                  const double outlet = p.problem == "backward_step" ? 5. : 1.;
                  if (std::abs(center[0]-outlet) < 1.e-12) boundary->set_boundary_id(2);
                }
            }
    triangulation.refine_global(p.initial_refinements);
  }

  void Problem::setup()
  {
    dofs.distribute_dofs(fe);
    DoFRenumbering::component_wise(dofs, std::vector<unsigned int>{0, 0, 1});
    owned = dofs.locally_owned_dofs();
    relevant = DoFTools::extract_locally_relevant_dofs(dofs);
    const auto counts = DoFTools::count_dofs_per_fe_block(dofs, std::vector<unsigned int>{0, 0, 1});
    n_velocity = counts[0];
    owned_blocks = {owned.get_view(0, counts[0]), owned.get_view(counts[0], dofs.n_dofs())};
    relevant_blocks = {relevant.get_view(0, counts[0]), relevant.get_view(counts[0], dofs.n_dofs())};

    constraints.clear();
    constraints.reinit(owned, relevant);
    DoFTools::make_hanging_node_constraints(dofs, constraints);
    const BoundaryValues boundary(p);
    VectorTools::interpolate_boundary_values(dofs, 0, boundary, constraints, fe.component_mask(velocity));
    if (!p.closed_boundary())
      VectorTools::interpolate_boundary_values(dofs, 1, boundary, constraints, fe.component_mask(velocity));
    // A natural outlet fixes pressure. Pin exactly one pressure DoF only for
    // full velocity Dirichlet problems; normalize its output after the solve.
    if (p.closed_boundary() && relevant.is_element(n_velocity)) constraints.add_line(n_velocity);
    constraints.close();

    pressure_constraints.clear();
    pressure_constraints.reinit(owned_blocks[1], relevant_blocks[1]);
    if (p.closed_boundary() && relevant_blocks[1].is_element(0)) pressure_constraints.add_line(0);
    pressure_constraints.close();

    BlockDynamicSparsityPattern graph(2, 2);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        graph.block(i, j).reinit(counts[i], counts[j], relevant_blocks[i]);
    graph.collect_sizes();
    // The p-p graph is also used by Q. Its entries in the physical operator
    // remain exactly zero, apart from the single pressure constraint row.
    DoFTools::make_sparsity_pattern(dofs, graph, constraints, false);
    SparsityTools::distribute_sparsity_pattern(graph, owned, comm, relevant);
    system_matrix.reinit(owned_blocks, graph, comm);
    pressure_mass.reinit(owned_blocks[1], owned_blocks[1], graph.block(1, 1), comm);

    solution.reinit(owned_blocks, comm);
    rhs.reinit(owned_blocks, comm);
    assembly_rhs.reinit(owned_blocks, relevant_blocks, comm);
    ghosted_solution.reinit(owned_blocks, relevant_blocks, comm);
  }

  void Problem::assemble()
  {
    const QGauss<2> quadrature(p.pressure_degree+2);
    FEValues<2> values(fe, quadrature, update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    std::vector<unsigned int> pressure_local;
    for (unsigned int i = 0; i < n; ++i)
      if (fe.system_to_component_index(i).first == 2) pressure_local.push_back(i);
    FullMatrix<double> cell_matrix(n, n), cell_mass(pressure_local.size(), pressure_local.size());
    dealii::Vector<double> cell_rhs(n);
    std::vector<types::global_dof_index> indices(n), pressure_indices(pressure_local.size());
    std::vector<Tensor<2, 2>> gradients(n);
    std::vector<double> divergences(n), pressures(n);
    assembly_rhs = 0.;
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell);
          cell_matrix = 0.;
          cell_mass = 0.;
          cell_rhs = 0.;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              for (unsigned int i = 0; i < n; ++i)
                {
                  gradients[i] = values[velocity].gradient(i, q);
                  divergences[i] = values[velocity].divergence(i, q);
                  pressures[i] = values[pressure].value(i, q);
                }
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  cell_matrix(i, j) += (p.viscosity*scalar_product(gradients[i], gradients[j])
                                       -divergences[i]*pressures[j]-pressures[i]*divergences[j])*values.JxW(q);
              for (unsigned int i = 0; i < pressure_local.size(); ++i)
                for (unsigned int j = 0; j < pressure_local.size(); ++j)
                  cell_mass(i, j) += pressures[pressure_local[i]]*pressures[pressure_local[j]]*values.JxW(q)/p.viscosity;
            }
          cell->get_dof_indices(indices);
          constraints.distribute_local_to_global(cell_matrix, cell_rhs, indices, system_matrix, assembly_rhs);
          for (unsigned int i = 0; i < pressure_local.size(); ++i) pressure_indices[i] = indices[pressure_local[i]]-n_velocity;
          pressure_constraints.distribute_local_to_global(cell_mass, pressure_indices, pressure_mass);
        }
    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    assembly_rhs.compress(VectorOperation::add);
    rhs = assembly_rhs;
  }

  void Problem::solve(const unsigned int cycle)
  {
    const double before_setup = MPI_Wtime();
    BlockInverse inverse_velocity(system_matrix.block(0, 0), p.velocity, "Velocity");
    BlockInverse inverse_pressure(pressure_mass, p.pressure, "Pressure");
    BlockPreconditioner preconditioner(system_matrix, inverse_velocity, inverse_pressure,
                                     p.block_preconditioner, rhs);
    setup_seconds = Utilities::MPI::max(MPI_Wtime()-before_setup, comm);
    rhs_norm = rhs.l2_norm();
    const double tolerance = std::max(p.absolute_tolerance, p.relative_tolerance*rhs_norm);
    SolverControl control(p.max_iterations, tolerance, p.solver_trace, p.solver_trace);
    control.enable_history_data();
    SolverFGMRES<BlockVector> solver(control, typename SolverFGMRES<BlockVector>::AdditionalData(p.restart));
    solution = 0.;
    const double before_solve = MPI_Wtime();
    solver.solve(system_matrix, solution, rhs, preconditioner);
    solve_seconds = Utilities::MPI::max(MPI_Wtime()-before_solve, comm);
    outer_iterations = control.last_step();
    velocity_iterations = inverse_velocity.iterations;
    pressure_iterations = inverse_pressure.iterations;

    BlockVector residual;
    residual.reinit(rhs);
    system_matrix.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    residual_norm = residual.l2_norm();
    AssertThrow(std::isfinite(residual_norm) && residual_norm <= 10.*tolerance,
                ExcMessage("The recomputed residual exceeds ten times the requested FGMRES tolerance."));
    // Check the constrained algebraic system before restoring prescribed values
    // and changing the arbitrary pressure constant for visualization/errors.
    constraints.distribute(solution);
    ghosted_solution = solution;
    ghosted_solution.update_ghost_values();
    if (p.closed_boundary())
      {
        const double mean = VectorTools::compute_mean_value(dofs, QGauss<2>(p.pressure_degree+2), ghosted_solution, 2);
        for (const auto i : owned_blocks[1]) solution.block(1)[i] -= mean;
        ghosted_solution = solution;
        ghosted_solution.update_ghost_values();
      }
    pcout << "  FGMRES iterations: " << outer_iterations
          << "; true residual: " << residual_norm
          << "; inner iterations (u,p): " << velocity_iterations << ',' << pressure_iterations << '\n';

    if (p.write_history)
      rank_zero_io([&]() {
        const auto filename = p.output_directory + "/residuals-" + Utilities::int_to_string(cycle, 3) + ".csv";
        std::ofstream file(filename);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file << "iteration,fgmres_residual_estimate\n" << std::setprecision(17);
        const auto &history = control.get_history_data();
        for (unsigned int i = 0; i < history.size(); ++i) file << i << ',' << history[i] << '\n';
        file.close();
      });
  }

  void Problem::diagnostics(const unsigned int cycle)
  {
    // At least six points per direction also resolve the quartic benchmark
    // and its squared error when the trial space has low polynomial degree.
    const QGauss<2> quadrature(std::max(6u, p.pressure_degree+3));
    FEValues<2> values(fe, quadrature,
                       update_values | update_gradients | update_quadrature_points | update_JxW_values);
    std::vector<Tensor<1, 2>> u(quadrature.size());
    std::vector<Tensor<2, 2>> grad_u(quadrature.size());
    std::vector<double> pressure_values(quadrature.size());
    ExactSolution exact(p);
    double u_error = 0., u_h1_error = 0., p_error = 0., divergence = 0., pressure_integral = 0., volume = 0.;
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell);
          values[velocity].get_function_values(ghosted_solution, u);
          values[velocity].get_function_gradients(ghosted_solution, grad_u);
          values[pressure].get_function_values(ghosted_solution, pressure_values);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              const double weight = values.JxW(q);
              divergence += std::pow(trace(grad_u[q]), 2)*weight;
              pressure_integral += pressure_values[q]*weight;
              volume += weight;
              if (p.has_exact_solution())
                {
                  const auto x = values.quadrature_point(q);
                  for (unsigned int c = 0; c < 2; ++c)
                    {
                      u_error += std::pow(u[q][c]-exact.value(x, c), 2)*weight;
                      u_h1_error += (grad_u[q][c]-exact.gradient(x, c)).norm_square()*weight;
                    }
                  p_error += std::pow(pressure_values[q]-exact.value(x, 2), 2)*weight;
                }
            }
        }
    u_error = std::sqrt(Utilities::MPI::sum(u_error, comm));
    u_h1_error = std::sqrt(Utilities::MPI::sum(u_h1_error, comm));
    p_error = std::sqrt(Utilities::MPI::sum(p_error, comm));
    divergence = std::sqrt(Utilities::MPI::sum(divergence, comm));
    const double mean_pressure = Utilities::MPI::sum(pressure_integral, comm)/Utilities::MPI::sum(volume, comm);

    const QGauss<1> face_quadrature(std::max(6u, p.pressure_degree+3));
    FEFaceValues<2> face_values(fe, face_quadrature, update_values | update_normal_vectors | update_JxW_values);
    std::vector<Tensor<1, 2>> face_u(face_quadrature.size());
    double inlet_flux = 0., outlet_flux = 0., boundary_flux = 0.;
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary())
            {
              face_values.reinit(cell, face);
              face_values[velocity].get_function_values(ghosted_solution, face_u);
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  const double flux = (face_u[q]*face_values.normal_vector(q))*face_values.JxW(q);
                  boundary_flux += flux;
                  if (cell->face(face)->boundary_id() == 1) inlet_flux += flux;
                  if (cell->face(face)->boundary_id() == 2) outlet_flux += flux;
                }
            }
    inlet_flux = Utilities::MPI::sum(inlet_flux, comm);
    outlet_flux = Utilities::MPI::sum(outlet_flux, comm);
    boundary_flux = Utilities::MPI::sum(boundary_flux, comm);
    pcout << "  ||div u||_L2: " << divergence << "; net boundary flux: " << boundary_flux << '\n';
    if (p.has_exact_solution())
      pcout << "  Errors: ||u-u_exact||_L2=" << u_error
            << ", |u-u_exact|_H1=" << u_h1_error << ", ||p-p_exact||_L2=" << p_error << '\n';

    rank_zero_io([&]() {
      summary << cycle << ',' << p.backend << ',' << p.problem << ',' << p.lid << ','
              << Utilities::MPI::n_mpi_processes(comm) << ',' << triangulation.n_global_active_cells() << ','
              << dofs.n_dofs() << ',' << p.pressure_degree+1 << ',' << p.pressure_degree << ',' << p.viscosity << ','
              << p.block_preconditioner << ',' << outer_iterations << ',' << velocity_iterations << ',' << pressure_iterations << ','
              << residual_norm << ',' << (rhs_norm == 0. ? residual_norm : residual_norm/rhs_norm) << ',' << divergence << ',';
      if (p.has_exact_solution()) summary << u_error << ',' << u_h1_error << ',' << p_error;
      else summary << "nan,nan,nan";
      summary << ',' << mean_pressure << ',' << inlet_flux << ',' << outlet_flux << ',' << boundary_flux << ','
              << setup_seconds << ',' << solve_seconds << '\n';
      summary.flush();
    });
    AssertThrow(p.maximum_velocity_error < 0. || u_error <= p.maximum_velocity_error,
                ExcMessage("Velocity L2 verification threshold exceeded."));
    AssertThrow(p.maximum_pressure_error < 0. || p_error <= p.maximum_pressure_error,
                ExcMessage("Pressure L2 verification threshold exceeded."));
    AssertThrow(p.maximum_divergence < 0. || divergence <= p.maximum_divergence,
                ExcMessage("Divergence L2 verification threshold exceeded."));
  }

  void Problem::output(const unsigned int cycle) const
  {
    if (!p.write_visualization) return;
    DataOut<2> data;
    data.attach_dof_handler(dofs);
    const std::vector<std::string> names{"velocity", "velocity", "pressure"};
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(3, DataComponentInterpretation::component_is_part_of_vector);
    interpretation[2] = DataComponentInterpretation::component_is_scalar;
    data.add_data_vector(ghosted_solution, names, DataOut<2>::type_dof_data, interpretation);
    data.build_patches(p.pressure_degree+1);
    data.write_vtu_with_pvtu_record(p.output_directory+"/", "stokes", cycle, comm, 3);
  }

  void Problem::run(const ParameterHandler &parameters)
  {
    rank_zero_io([&]() {
      std::filesystem::create_directories(p.output_directory);
      std::ofstream used_parameters(p.output_directory+"/parameters-used.prm");
      used_parameters.exceptions(std::ios::failbit | std::ios::badbit);
      parameters.print_parameters(used_parameters, ParameterHandler::PRM);
      used_parameters.close();
      summary.exceptions(std::ios::failbit | std::ios::badbit);
      summary.open(p.output_directory+"/summary.csv");
      summary << std::setprecision(17)
              << "cycle,backend,problem,cavity_lid,mpi_ranks,cells,dofs,velocity_degree,pressure_degree,viscosity,block_preconditioner,outer_iterations,velocity_inner_iterations,pressure_inner_iterations,true_residual,relative_true_residual,divergence_L2,velocity_L2_error,velocity_H1_seminorm_error,pressure_L2_error,mean_pressure,inlet_outward_flux,outlet_outward_flux,net_outward_flux,preconditioner_setup_seconds,solve_seconds\n";
    });
    pcout << std::setprecision(10) << "Backend: " << p.backend << "; available: " << StokesLA::available_backends()
          << "\nProblem: " << p.problem << "; Taylor-Hood Q" << p.pressure_degree+1 << "/Q" << p.pressure_degree
          << "; block preconditioner: " << p.block_preconditioner << '\n';
    make_grid();
    for (unsigned int cycle = 0; cycle < p.cycles; ++cycle)
      {
        if (cycle != 0) triangulation.refine_global(1);
        setup();
        pcout << "Cycle " << cycle << ": " << triangulation.n_global_active_cells() << " cells, " << dofs.n_dofs() << " DoFs\n";
        assemble();
        solve(cycle);
        diagnostics(cycle);
        output(cycle);
      }
  }
} // namespace Stokes

int main(int argc, char **argv)
{
  // Keep MPI alive through the catch block. A failure confined to one rank
  // must abort its peers before MPI_Finalize can wait for them indefinitely.
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  try
    {
      using namespace dealii;
      const MPI_Comm comm = MPI_COMM_WORLD;
      const bool rank_zero = Utilities::MPI::this_mpi_process(comm) == 0;
      Stokes::Settings settings;
      ParameterHandler prm;
      settings.declare_parameters(prm);

      if (argc == 2 && std::string(argv[1]) == "--help")
        {
          if (rank_zero)
            std::cout << "Usage: stokes [setup.prm]\n"
                      << "With no argument, create parameters.prm and exit; an existing file is preserved.\n"
                      << "A missing explicitly named file is created with documented defaults and the program exits.\n"
                      << "Available backends: " << StokesLA::available_backends() << '\n';
          return 0;
        }
      AssertThrow(argc <= 2, ExcMessage("Usage: stokes [setup.prm]"));
      const std::string filename = argc == 2 ? argv[1] : "parameters.prm";
      std::string error, contents;
      bool generate = argc == 1;
      if (rank_zero)
        {
          try
            {
              const bool exists = std::filesystem::exists(filename);
              generate = generate || !exists;
              if (generate)
                {
                  if (exists)
                    std::cout << "Preserving existing " << filename << ". Pass it explicitly to run.\n";
                  else
                    {
                      std::ofstream output(filename);
                      output.exceptions(std::ios::failbit | std::ios::badbit);
                      prm.print_parameters(output, ParameterHandler::PRM);
                      output.close();
                      std::cout << "Wrote " << filename << ". Edit it and run: " << argv[0] << ' ' << filename << '\n';
                    }
                }
              else
                {
                  std::ifstream input(filename);
                  input.exceptions(std::ios::badbit);
                  AssertThrow(input.is_open(), ExcMessage("Cannot open " + filename));
                  std::ostringstream buffer;
                  buffer << input.rdbuf();
                  contents = buffer.str();
                }
            }
          catch (const std::exception &exception) { error = exception.what(); }
        }
      error = Utilities::MPI::broadcast(comm, error, 0);
      AssertThrow(error.empty(), ExcMessage(error));
      generate = Utilities::MPI::broadcast(comm, generate, 0);
      if (generate) return 0;
      contents = Utilities::MPI::broadcast(comm, contents, 0);
      prm.parse_input_from_string(contents);
      const auto parameter_directory = std::filesystem::absolute(filename).parent_path();
      settings.velocity.preconditioner.psc.resolve_paths(parameter_directory);
      settings.pressure.preconditioner.psc.resolve_paths(parameter_directory);
      settings.validate();
      if (rank_zero && settings.solver_trace) deallog.depth_console(2);
      Stokes::Problem problem(settings);
      problem.run(prm);
      return 0;
    }
  catch (const std::exception &exception)
    {
      std::cerr << "Stokes example failed: " << exception.what() << '\n';
      if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
}
