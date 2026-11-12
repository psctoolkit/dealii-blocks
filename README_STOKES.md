# Stokes block-solver example

`stokes` is a two-dimensional, MPI-distributed deal.II example for comparing
PETSc, Trilinos, and PSCToolkit matrix and preconditioner implementations. It
solves the reference problems from the classical book "Elman, Howard C., David J. 
Silvester, and Andrew J. Wathen. Finite elements and fast iterative solvers: 
with applications in incompressible fluid dynamics. Oxford university press, 2005.", 
including all three driven-cavity boundary conventions. The implementation consists of
[stokes.cc](stokes.cc), which contains the discretization and block solvers,
and [stokes_backends.h](stokes_backends.h), which contains the backend adapters.

## Build and run

Use a deal.II installation with MPI, p4est, and at least one supported algebra
backend. The PSCToolkit backend requires a compatible deal.II build with
`DEAL_II_WITH_PSBLAS` and `DEAL_II_WITH_AMG4PSBLAS`. The repository's dependency
submodules and existing configuration scripts identify the intended versions.
PETSc must use real scalars. Trilinos uses its Epetra interfaces; the `AMG`
preconditioner additionally needs ML.

The native dependency configuration names are unchanged:

```text
-DDEAL_II_WITH_PSBLAS=ON -DPSBLAS_DIR=/path/to/psblas3
-DDEAL_II_WITH_AMG4PSBLAS=ON -DAMG4PSBLAS_DIR=/path/to/amg4psblas
```

These switches configure deal.II. The example inherits those dependencies
through `deal_ii_setup_target`; it introduces no replacement directory or
feature switches. The rectangular PSBLAS C interface must match the versions
pinned by this repository's submodules.

```sh
cmake -S . -B build-stokes \
  -DDEAL_II_DIR=/path/to/dealii/install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-stokes --target stokes -j
```

The executable selects a compiled backend at run time. Changing a parameter
cannot enable a library that was absent when deal.II was built. `--help`
reports the available backends.

```sh
./build-stokes/stokes --help
./build-stokes/stokes
```

With no argument, the program writes a fully documented `parameters.prm` in
the working directory and exits. If that file already exists, it preserves
the file and exits. Pass the filename explicitly to solve:

```sh
mpirun -np 4 ./build-stokes/stokes parameters.prm
```

An explicitly named file that does not exist is also generated with defaults;
the program then exits so that it can be edited. Only rank zero reads or
writes the main parameter file, and its contents are broadcast to the other
ranks. The generated file declares the choices for every backend and both
block solvers. A shorter input file may specify only the values to override.

Small examples are provided in [parameters](parameters):

| File | Test case |
| --- | --- |
| [poiseuille.prm](parameters/poiseuille.prm) | Channel with an analytic solution |
| [backward_step.prm](parameters/backward_step.prm) | Backward-facing step |
| [cavity_regularized.prm](parameters/cavity_regularized.prm) | Smooth lid profile |
| [cavity_leaky.prm](parameters/cavity_leaky.prm) | Unit lid velocity including the top corners |
| [cavity_watertight.prm](parameters/cavity_watertight.prm) | Unit lid velocity with zero top-corner values |
| [colliding.prm](parameters/colliding.prm) | Polynomial analytic solution |

For example:

```sh
mpirun -np 4 ./build-stokes/stokes parameters/colliding.prm
```

Use a different `Output/Directory` for each experiment. Each run writes its
summary and cycle files to that directory, replacing files of the same name.

## Discretization and benchmark definitions

The equations are

\[
-\nu\Delta\boldsymbol u+\nabla p=0,
\qquad \nabla\cdot\boldsymbol u=0,
\qquad \nu>0.
\]

The velocity bilinear form is
\(\nu(\nabla\boldsymbol u,\nabla\boldsymbol v)\). Its natural outlet condition
is

\[
\nu\,\partial_n\boldsymbol u-p\boldsymbol n=0.
\]

This is the vector-Laplacian formulation used by the excerpt's outlet
condition (5.13). A symmetric-gradient stress would give a different natural
outlet problem.

Only continuous Taylor–Hood elements on quadrilateral meshes are generated:

\[
V_h\times Q_h=[Q_{k+1}]^2\times Q_k,\qquad k\ge1.
\]

`Discretization/Pressure degree` is `k`; the velocity degree is always `k+1`.
The square starts with a 2 by 2 mesh. The step starts with eleven unit-square
cells. `Initial global refinements` refines this mesh before the first solve;
each additional `Refinement cycles` iteration uniformly refines once more.
Every cycle assembles a new system and starts its solve from zero. There is
no equal-order option, stabilization parameter, or pressure penalty.

### Poiseuille channel: Example 5.1.1, pages 217–218

Set `Problem/Test case = poiseuille`. On \(\Omega=(-1,1)^2\),

\[
u_x=1-y^2,\qquad u_y=0,\qquad p=2\nu(1-x).
\]

The left inlet has this prescribed velocity, the top and bottom walls have
zero velocity, and the right boundary has the natural outlet condition.
The outlet fixes the pressure level, so no pressure constraint or subsequent
mean shift is applied. At \(\nu=1\), these are equations (5.12)–(5.13) in the
excerpt. Both fields are representable by the default \(Q_2/Q_1\) pair.

### Backward-facing step: Example 5.1.2, page 219

Set `Problem/Test case = backward_step`. The domain is
\((-1,5)\times(-1,1)\) with its lower-left unit square removed; the reentrant
corner is \((0,0)\). The inlet is \(x=-1,\ 0\le y\le1\), with

\[
u_x=4U y(1-y),\qquad u_y=0,
\]

where `Problem/Step inlet peak velocity` sets \(U\). Its default is one.
The excerpt specifies a parabolic profile without fixing its amplitude;
this normalization is therefore stated explicitly. The outlet is
\(x=5,\ -1<y<1\); all remaining walls have zero velocity. The outlet condition
again fixes the pressure level.

There is no analytic solution in the excerpt. Its reentrant-corner pressure
is singular, so this case reports solver diagnostics and fluxes without
inventing an exact-solution error.

### Driven cavity: Example 5.1.3, pages 220–221

Set `Problem/Test case = cavity`. The domain is \((-1,1)^2\). The lid is
\(y=1\), with zero vertical velocity. Choose `Problem/Cavity lid`:

| Value | Horizontal lid velocity | Values at \((\pm1,1)\) |
| --- | --- | --- |
| `leaky` | \(1\) | \(1\) |
| `watertight` | \(1\) in the open lid interval | \(0\) |
| `regularized` | \(1-x^4\) | \(0\) |

The remaining boundary has zero prescribed velocity, subject to the leaky
top-corner convention. One boundary function supplies all boundary values,
so face traversal order cannot change the corner choice. For the leaky
convention, the nonzero corner values also affect the adjacent finite-element
wall traces. The unregularized lids have corner singularities.

One pressure degree of freedom is pinned during the algebraic solve. After
checking the algebraic residual, the program shifts the pressure to zero
global mean for output. This benchmark has no supplied analytic solution.

### Colliding flow: Example 5.1.4, page 221

Set `Problem/Test case = colliding`. On \((-1,1)^2\),

\[
u_x=20xy^3,\qquad u_y=5x^4-5y^4,
\qquad p=\nu(60x^2y-20y^3).
\]

The finite-element interpolant of this velocity is prescribed on the entire
boundary. These fields solve the homogeneous equations above; multiplication
of the excerpt's pressure by \(\nu\) keeps them valid when viscosity changes.
The analytic pressure has zero mean on the square. A pressure degree of
freedom is pinned for the solve, followed by the same zero-mean normalization
as the cavity before errors are evaluated.

The default \(Q_2/Q_1\) pair has discretization error for this problem.
\(Q_4/Q_3\), selected with `Pressure degree = 3`, contains the exact fields.
The error quadrature resolves the quartic analytic velocity even when a lower
finite-element degree is used.

## Block preconditioners and auxiliary solves

The assembled matrix is

\[
K=\begin{bmatrix}A&B^T\\B&0\end{bmatrix},
\qquad
A_{ij}=\nu(\nabla\boldsymbol\phi_j,\nabla\boldsymbol\phi_i),
\qquad
B_{ij}=-(\psi_i,\nabla\cdot\boldsymbol\phi_j).
\]

The pressure mass approximation is assembled separately:

\[
Q=\frac1\nu M_p\approx BA^{-1}B^T.
\]

The physical pressure-pressure block stays zero, apart from the algebraic
gauge row in a closed-domain problem. \(Q\) is used only by the
preconditioner. `Linear algebra/Block preconditioner` selects:

| Value | Approximate block factor whose inverse is applied |
| --- | --- |
| `diagonal` | \(\operatorname{diag}(A,Q)\) |
| `upper triangular` | \(\begin{bmatrix}A&B^T\\0&-Q\end{bmatrix}\) |
| `lower triangular` | \(\begin{bmatrix}A&0\\B&-Q\end{bmatrix}\) |

The minus sign in a triangular pressure diagonal matches the Schur
complement \(-BA^{-1}B^T\). The diagonal option uses the positive pressure
mass block.

For example, the upper triangular application to \((r_u,r_p)\) is

\[
z_p=-\mathcal I_Q r_p,
\qquad
z_u=\mathcal I_A(r_u-B^Tz_p),
\]

where each \(\mathcal I\) is independently configurable:

- `Action = apply` applies the selected backend preconditioner once. Its own
  native sweep/cycle settings still determine the work inside that action.
- `Action = solve` starts an auxiliary solve from zero, using that same
  backend preconditioner. The velocity system uses \(A\); the pressure
  auxiliary system uses \(Q\).

Both `Velocity block` and `Pressure block` provide `Solver = cg|gmres|fgmres`,
`Maximum iterations`, `Restart length`, and stopping tolerances. The inner
target is

\[
\max(\text{absolute tolerance},
     \text{relative tolerance}\,\lVert r_{\rm block}\rVert_2).
\]

Inner failure is reported as an error. Use CG only when the chosen block
preconditioner is symmetric positive definite; an arbitrary ILU, Schwarz,
or multigrid setup need not satisfy that condition.

The outer method is always deal.II FGMRES. This accommodates block actions
whose auxiliary iteration count changes between applications. Its own
tolerances and basis length are under `Linear algebra/FGMRES`. The executable
recomputes \(\lVert b-Kx\rVert_2\) and rejects a result exceeding ten times
the requested outer target. This check occurs before restoring boundary
values and changing the closed-domain pressure constant.

## Choosing a backend and its native options

```text
subsection Linear algebra
  set Backend = PSBLAS
  set Block preconditioner = upper triangular

  subsection Velocity block
    set Action = solve
    set Solver = fgmres
    set Relative tolerance = 1e-2
    subsection AMG4PSBLAS
      set ptype = AS
      set SMOOTHER_SWEEPS = 1
      set SMOOTHER_STOP = FALSE
      set SUB_SOLVE = ILU
      set SUB_OVR = 1
      set SUB_RESTR = HALO
      set SUB_PROL = SUM
      set SUB_FILLIN = 0
    end
  end

  subsection Pressure block
    set Action = apply
    subsection AMG4PSBLAS
      set ptype = JACOBI
    end
  end
end
```

`Backend = PSBLAS` selects PSCToolkit. Preconditioner options are under the
native `AMG4PSBLAS` name. `ptype` is passed to `amg_c_dprecinit`; values such
as `ML`, `AS`, `BJAC`, `JACOBI`, `L1-JACOBI`, `POLY`, and `NONE` keep their
native meaning. Named settings such as `SMOOTHER_TYPE`, `SUB_SOLVE`,
`SUB_FILLIN`, `ML_CYCLE`, and `COARSE_SOLVE` are forwarded without renaming
them or expanding hidden presets. Empty entries retain library defaults.
Native availability and admissible combinations follow the linked
AMG4PSBLAS build.

`PRE`, `POST`, and `COARSE` subsections expose positioned settings. The
generated file also documents `Additional options` and `Options file` for
typed native entries, including explicit level ranges and indexed settings.
For example:

```text
set Additional options = hierarchy integer MAX_LEVS 10; smoother string SUB_SOLVE ILU position=PRE
```

The general entry form is

```text
hierarchy|smoother integer|real|string NAME VALUE [level=N|N:M|coarse] [position=PRE|POST] [index=N]
```

`hierarchy` and `smoother` specify the native construction phase. Values with
spaces must be quoted. Entries in `Additional options` follow the named
settings; an `Options file` is applied afterward. Its relative path is
resolved against the main parameter file's directory, and its contents are
read on rank zero and broadcast. The names and values themselves remain
native. `afmt` and `chfmt` expose the corresponding native format arguments;
`Verbosity` prints AMG4PSBLAS's description after setup.

The equivalent outer controls apply when `Backend = PETSc` or `Trilinos`:

| Backend | Per-block subsection | Selection and further settings |
| --- | --- | --- |
| PETSc | `PETSc` | Native `pc_type`, plus an `Options` string such as `-pc_asm_overlap 1 -sub_pc_type ilu` |
| Trilinos | `Trilinos` | `Preconditioner = AMG|Jacobi|SSOR|ILU|ILUT|IC|Identity`, with the declared ML/Ifpack settings |

Each PETSc block has its own options database, so its options do not change
the other block. PETSc preconditioner availability, including third-party
packages and parallel direct solvers, depends on that PETSc installation.

The default backend is PSBLAS when available, otherwise PETSc, otherwise
Trilinos. The default velocity preconditioner is AMG (`ML`, `gamg`, or ML
`AMG`); the default pressure preconditioner is Jacobi. Both default block
actions are auxiliary FGMRES solves. The generated parameter file is the
authoritative inventory of defaults for the executable being used.

For one-level PSCToolkit experiments select a native type such as `AS`,
`BJAC`, or `JACOBI`. The pinned multilevel builder's `ML` with `MAX_LEVS=1`
is rejected explicitly. The adapter also rejects known misrouted
level-specific string setters in that pinned version instead of silently
changing the requested algorithm.

## Output and checks

Every run writes `parameters-used.prm` with the complete effective parameter
set and `summary.csv`, with one row per refinement cycle. Use distinct output
directories when comparing runs; a repeated directory replaces its CSV and
parameter records. The summary records
the problem, backend, MPI rank count, mesh and finite-element sizes, block
form, outer and accumulated inner iteration counts, true algebraic residual,
divergence norm, pressure mean, boundary fluxes, and setup/solve times.
Times are maxima over MPI ranks. Preconditioner setup time excludes mesh
generation and finite-element assembly.

For the analytic problems, the summary also includes velocity \(L^2\)
error, velocity \(H^1\) seminorm error, and pressure \(L^2\) error. Other
problems write `nan` in those error columns. Fluxes use outward normals;
inlet flux is therefore normally negative. For closed-boundary problems,
the separate inlet/outlet columns are zero and the net boundary flux is
still evaluated.

`Output/Write residual history` controls the per-cycle CSV files containing
outer FGMRES residual estimates. `Output/Write visualization` controls
parallel VTU/PVTU output of the velocity and pressure fields. Open the PVTU
file in ParaView to view all MPI partitions together.

`Verification` can set nonnegative maximum velocity and pressure \(L^2\)
errors for the two analytic cases, and a maximum divergence \(L^2\) norm
for any case. Negative thresholds disable those checks. Taylor–Hood
velocities satisfy the discrete incompressibility constraint weakly; their
pointwise divergence generally does not vanish exactly.

This example uses uniform refinement and solves the steady velocity-pressure
system. It does not add an auxiliary streamfunction solve. The excerpt's
streamfunction plots are postprocessing examples, and velocity/pressure
output is sufficient to inspect the block-solver experiments. No scaling or
performance conclusion should be drawn without recording the compiler,
library versions, machine, rank count, parameters, and numerical errors for
the compared runs.