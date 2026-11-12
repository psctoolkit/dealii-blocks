cmake_minimum_required(VERSION 3.16)

# Keep the native deal.II dependency switches and directory names. Configure
# those in deal.II itself; this example uses the capabilities of that build.
find_package(deal.II REQUIRED
  HINTS ${DEAL_II_DIR} ../build ../dealii-build $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(dealii_blocks LANGUAGES CXX)

if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR
    "Stokes requires deal.II configured with DEAL_II_WITH_MPI=ON and DEAL_II_WITH_P4EST=ON.")
endif()

set(stokes_backends)
if(DEAL_II_WITH_PETSC AND NOT DEAL_II_PETSC_WITH_COMPLEX)
  list(APPEND stokes_backends PETSc)
endif()
# Epetra and ML were mandatory in supported pre-9.7 Trilinos builds.
if(DEAL_II_WITH_TRILINOS)
  if(DEAL_II_PACKAGE_VERSION VERSION_LESS 9.7.0 OR
     (DEAL_II_TRILINOS_WITH_EPETRA AND DEAL_II_TRILINOS_WITH_ML))
    list(APPEND stokes_backends Trilinos)
  endif()
endif()
if(DEAL_II_WITH_PSBLAS AND DEAL_II_WITH_AMG4PSBLAS)
  list(APPEND stokes_backends PSBLAS)
endif()
if(NOT stokes_backends)
  message(FATAL_ERROR
    "Enable a real-valued PETSc backend, Trilinos with Epetra and ML, or both "
    "DEAL_II_WITH_PSBLAS=ON and DEAL_II_WITH_AMG4PSBLAS=ON in deal.II. "
    "The native PSCToolkit discovery variables remain PSBLAS_DIR and AMG4PSBLAS_DIR.")
endif()
message(STATUS "Stokes runtime backends: ${stokes_backends}")

add_executable(stokes stokes.cc)
target_compile_features(stokes PRIVATE cxx_std_17)
deal_ii_setup_target(stokes)

include(CTest)
if(BUILD_TESTING)
  set(stokes_test_root "${CMAKE_CURRENT_BINARY_DIR}/tests")
  file(MAKE_DIRECTORY "${stokes_test_root}")

  # Each test has its own parameter file/output directory so ctest -j is safe.
  # Jacobi needs no optional direct solver package; the outer flexible solve
  # still exercises native PC application and genuinely nested Krylov solves.
  function(stokes_numerical_test name backend problem degree refinements nu form u_action p_action ranks)
    set(test_dir "${stokes_test_root}/${name}")
    file(MAKE_DIRECTORY "${test_dir}")
    set(prm "${test_dir}/setup.prm")
    file(WRITE "${prm}"
"subsection Problem
  set Test case = ${problem}
  set Viscosity = ${nu}
end
subsection Discretization
  set Pressure degree = ${degree}
  set Initial global refinements = ${refinements}
end
subsection Linear algebra
  set Backend = ${backend}
  set Block preconditioner = ${form}
  subsection FGMRES
    set Maximum iterations = 500
    set Restart length = 150
    set Relative tolerance = 1e-11
    set Absolute tolerance = 1e-13
  end
  subsection Velocity block
    set Action = ${u_action}
    set Relative tolerance = 1e-3
    subsection PETSc
      set pc_type = jacobi
    end
    subsection Trilinos
      set Preconditioner = Jacobi
    end
    subsection AMG4PSBLAS
      set ptype = JACOBI
      set SMOOTHER_SWEEPS = 1
    end
  end
  subsection Pressure block
    set Action = ${p_action}
    set Relative tolerance = 1e-3
    subsection PETSc
      set pc_type = jacobi
    end
    subsection Trilinos
      set Preconditioner = Jacobi
    end
    subsection AMG4PSBLAS
      set ptype = JACOBI
      set SMOOTHER_SWEEPS = 1
    end
  end
end
subsection Output
  set Directory = output
  set Write visualization = false
  set Write residual history = false
end
subsection Verification
  set Maximum velocity L2 error = 1e-6
  set Maximum pressure L2 error = 1e-6
  set Maximum divergence L2 norm = 1e-6
end
")
    if(ranks GREATER 1)
      add_test(NAME ${name}
        COMMAND ${DEAL_II_MPIEXEC} ${DEAL_II_MPIEXEC_NUMPROC_FLAG} ${ranks}
          ${DEAL_II_MPIEXEC_PREFLAGS} $<TARGET_FILE:stokes>
          ${DEAL_II_MPIEXEC_POSTFLAGS} "${prm}")
    else()
      add_test(NAME ${name} COMMAND stokes "${prm}")
    endif()
    set_tests_properties(${name} PROPERTIES
      WORKING_DIRECTORY "${test_dir}" TIMEOUT 180 PROCESSORS ${ranks}
      LABELS "stokes;${backend};exact-solution")
  endfunction()

  foreach(backend IN LISTS stokes_backends)
    string(TOLOWER "${backend}" key)
    # Poiseuille Q2/Q1 checks outlet traction and the absolute pressure level.
    stokes_numerical_test(${key}_poiseuille_upper ${backend} poiseuille 1 1 1
      "upper triangular" solve solve 1)
    stokes_numerical_test(${key}_poiseuille_diagonal ${backend} poiseuille 1 1 1
      diagonal apply apply 1)
    stokes_numerical_test(${key}_poiseuille_lower ${backend} poiseuille 1 1 0.2
      "lower triangular" apply solve 1)
    # Q4/Q3 contains the colliding polynomial exactly; nu!=1 checks scaling.
    stokes_numerical_test(${key}_colliding ${backend} colliding 3 0 0.3
      "upper triangular" solve apply 1)
    if(DEAL_II_MPIEXEC)
      stokes_numerical_test(${key}_poiseuille_mpi2 ${backend} poiseuille 1 1 1
        "upper triangular" solve solve 2)
    endif()
  endforeach()

  # Check both default generation and preservation of a user's existing file.
  file(MAKE_DIRECTORY "${stokes_test_root}/defaults")
  set(default_check [=[
set(work "@stokes_test_root@/defaults")
file(REMOVE "${work}/parameters.prm")
execute_process(COMMAND "$<TARGET_FILE:stokes>" WORKING_DIRECTORY "${work}" RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR NOT EXISTS "${work}/parameters.prm")
  message(FATAL_ERROR "No-argument invocation did not generate parameters.prm")
endif()
file(READ "${work}/parameters.prm" contents)
foreach(option "MAX_LEVS" "SUB_SOLVE" "pc_type" "Pressure degree" "Block preconditioner")
  string(FIND "${contents}" "${option}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "Generated defaults are missing ${option}")
  endif()
endforeach()
file(APPEND "${work}/parameters.prm" "\n# preservation sentinel\n")
file(SHA256 "${work}/parameters.prm" before)
execute_process(COMMAND "$<TARGET_FILE:stokes>" WORKING_DIRECTORY "${work}" RESULT_VARIABLE status)
file(SHA256 "${work}/parameters.prm" after)
if(NOT status EQUAL 0 OR NOT before STREQUAL after)
  message(FATAL_ERROR "No-argument invocation modified an existing parameter file")
endif()
]=])
  string(CONFIGURE "${default_check}" default_check @ONLY)
  file(GENERATE OUTPUT "${stokes_test_root}/check-defaults.cmake" CONTENT "${default_check}")
  add_test(NAME default_parameters
    COMMAND ${CMAKE_COMMAND} -P "${stokes_test_root}/check-defaults.cmake")
  set_tests_properties(default_parameters PROPERTIES LABELS "stokes;parameters" TIMEOUT 30)
endif()
