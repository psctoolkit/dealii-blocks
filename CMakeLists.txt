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

# deal.II exports the AMG4PSBLAS -> PSBLAS dependency and its link ordering.
# Complete only the optional MUMPS precisions needed by the extended AMG C
# bindings: one object contains real and complex routines, whereas deal.II's
# MUMPS interface normally exports only the double-precision solver.
if(DEAL_II_WITH_PSBLAS AND DEAL_II_WITH_AMG4PSBLAS)
  get_target_property(_stokes_amg_includes
    dealii::interface_amg4psblas INTERFACE_INCLUDE_DIRECTORIES)
  unset(_stokes_amg_config CACHE)
  find_file(_stokes_amg_config amg_config.h
    PATHS ${_stokes_amg_includes} NO_DEFAULT_PATH)
  if(NOT _stokes_amg_config)
    message(FATAL_ERROR "Cannot find amg_config.h in the AMG4PSBLAS include directories exported by deal.II.")
  endif()
  file(STRINGS "${_stokes_amg_config}" _stokes_amg_mumps
    REGEX "^[ \t]*#[ \t]*define[ \t]+AMG_HAVE_MUMPS([ \t]|$)")
  unset(_stokes_amg_config CACHE)

  if(_stokes_amg_mumps)
    if(NOT TARGET dealii::interface_mumps)
      message(FATAL_ERROR "AMG4PSBLAS was built with MUMPS; enable DEAL_II_WITH_MUMPS in deal.II.")
    endif()
    get_target_property(_stokes_mumps_libraries
      dealii::interface_mumps INTERFACE_LINK_LIBRARIES)
    set(_stokes_amg_dependencies)
    set(_stokes_mumps_library_dirs)
    foreach(_library IN LISTS _stokes_mumps_libraries)
      get_filename_component(_name "${_library}" NAME)
      if(IS_ABSOLUTE "${_library}" AND _name MATCHES "^(lib)?dmumps([.-]|$)")
        get_filename_component(_directory "${_library}" DIRECTORY)
        list(APPEND _stokes_mumps_library_dirs "${_directory}")
      endif()
    endforeach()
    # AMG4PSBLAS's native MUMPS interface uses all four scalar libraries.
    # Resolve the additional ones beside dmumps to keep the installation and
    # compiler/MPI ABI consistent with the library already chosen by deal.II.
    foreach(_precision s c z)
      unset(_stokes_mumps_precision_library CACHE)
      find_library(_stokes_mumps_precision_library NAMES ${_precision}mumps
        PATHS ${_stokes_mumps_library_dirs} NO_DEFAULT_PATH)
      if(NOT _stokes_mumps_precision_library)
        message(FATAL_ERROR "AMG4PSBLAS needs lib${_precision}mumps from the same MUMPS installation as dmumps. Check MUMPS_DIR in the deal.II configuration.")
      endif()
      list(APPEND _stokes_amg_dependencies "${_stokes_mumps_precision_library}")
    endforeach()
    unset(_stokes_mumps_precision_library CACHE)
    list(APPEND _stokes_amg_dependencies ${_stokes_mumps_libraries})
    set_property(TARGET dealii::interface_amg4psblas APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES "${_stokes_amg_dependencies}")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  set(stokes_test_root "${CMAKE_CURRENT_BINARY_DIR}/tests")
  file(MAKE_DIRECTORY "${stokes_test_root}")

  # Each test has its own parameter file/output directory so ctest -j is safe.
  # Jacobi needs no optional direct solver package. The FGMRES cases exercise
  # nested solves; GMRES/MINRES use fixed native preconditioner applications.
  function(stokes_numerical_test name backend problem degree refinements nu form u_action p_action ranks outer_method)
    cmake_parse_arguments(PARSE_ARGV 11 case "CHECK_OUTPUT;CHECK_MULTILEVEL"
      "LID;CYCLES;VELOCITY_PTYPE;VELOCITY_NATIVE;OPTIONS_FILE_CONTENT;LABELS" "")
    if(NOT DEFINED case_LID)
      set(case_LID regularized)
    endif()
    if(NOT DEFINED case_CYCLES)
      set(case_CYCLES 1)
    endif()
    if(NOT DEFINED case_VELOCITY_PTYPE)
      set(case_VELOCITY_PTYPE JACOBI)
    endif()
    if(case_CHECK_OUTPUT)
      set(write_output true)
    else()
      set(write_output false)
    endif()
    if(problem STREQUAL poiseuille OR problem STREQUAL colliding)
      set(verification
"  set Maximum velocity L2 error = 1e-6
  set Maximum pressure L2 error = 1e-6
  set Maximum divergence L2 norm = 1e-6")
      set(problem_label exact-solution)
    else()
      set(verification "")
      set(problem_label benchmark)
    endif()
    set(test_dir "${stokes_test_root}/${name}")
    file(MAKE_DIRECTORY "${test_dir}")
    set(run_directory "${test_dir}")
    if(DEFINED case_OPTIONS_FILE_CONTENT)
      file(MAKE_DIRECTORY "${test_dir}/options")
      file(WRITE "${test_dir}/options/velocity.options" "${case_OPTIONS_FILE_CONTENT}")
      # Exercise resolution relative to setup.prm rather than the process cwd.
      set(run_directory "${test_dir}/working")
      file(MAKE_DIRECTORY "${run_directory}")
    endif()
    set(prm "${test_dir}/setup.prm")
    file(WRITE "${prm}"
"subsection Problem
  set Test case = ${problem}
  set Cavity lid = ${case_LID}
  set Viscosity = ${nu}
end
subsection Discretization
  set Pressure degree = ${degree}
  set Initial global refinements = ${refinements}
  set Refinement cycles = ${case_CYCLES}
end
subsection Linear algebra
  set Backend = ${backend}
  set Block preconditioner = ${form}
  subsection LinearSolver
    set Method = ${outer_method}
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
      set smoother_sweeps = 1
    end
    subsection AMG4PSBLAS
      set ptype = ${case_VELOCITY_PTYPE}
      set SMOOTHER_SWEEPS = 1
${case_VELOCITY_NATIVE}
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
      set smoother_sweeps = 1
    end
    subsection AMG4PSBLAS
      set ptype = JACOBI
      set SMOOTHER_SWEEPS = 1
    end
  end
end
subsection Output
  set Directory = output
  set Write visualization = ${write_output}
  set Write residual history = ${write_output}
end
subsection Verification
${verification}
end
")
    if(ranks GREATER 1)
      set(run_command ${DEAL_II_MPIEXEC} ${DEAL_II_MPIEXEC_NUMPROC_FLAG} ${ranks}
          ${DEAL_II_MPIEXEC_PREFLAGS} $<TARGET_FILE:stokes>
          ${DEAL_II_MPIEXEC_POSTFLAGS} "${prm}")
    else()
      set(run_command $<TARGET_FILE:stokes> "${prm}")
    endif()

    # The test runner uses only CMake. Besides solver verification, it checks
    # the pressure gauge, refinement-cycle records, and requested output files.
    set(run_configuration "set(STOKES_RUN_COMMAND)\n")
    foreach(argument IN LISTS run_command)
      string(APPEND run_configuration
        "list(APPEND STOKES_RUN_COMMAND [==[${argument}]==])\n")
    endforeach()
    foreach(variable test_dir run_directory backend problem ranks case_CYCLES case_CHECK_OUTPUT case_CHECK_MULTILEVEL)
      string(APPEND run_configuration "set(${variable} [==[${${variable}}]==])\n")
    endforeach()
    string(APPEND run_configuration
      "include([==[${CMAKE_CURRENT_SOURCE_DIR}/tests/check_run.cmake]==])\n")
    file(GENERATE OUTPUT "${test_dir}/run.cmake" CONTENT "${run_configuration}")
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -P "${test_dir}/run.cmake")
    set_tests_properties(${name} PROPERTIES
      WORKING_DIRECTORY "${test_dir}" TIMEOUT 180 PROCESSORS ${ranks}
      LABELS "stokes;${backend};${outer_method};${problem_label};${case_LABELS}")
  endfunction()

  foreach(backend IN LISTS stokes_backends)
    string(TOLOWER "${backend}" key)
    # Poiseuille Q2/Q1 checks outlet traction and the absolute pressure level.
    stokes_numerical_test(${key}_poiseuille_upper ${backend} poiseuille 1 1 1
      "upper triangular" solve solve 1 FGMRES)
    stokes_numerical_test(${key}_poiseuille_diagonal ${backend} poiseuille 1 1 1
      diagonal apply apply 1 FGMRES)
    stokes_numerical_test(${key}_poiseuille_lower ${backend} poiseuille 1 1 0.2
      "lower triangular" apply solve 1 FGMRES)
    # Q4/Q3 contains the colliding polynomial exactly; nu!=1 checks scaling.
    stokes_numerical_test(${key}_colliding ${backend} colliding 3 0 0.3
      "upper triangular" solve apply 1 FGMRES)
    # Fixed upper-triangular application is suitable for ordinary GMRES.
    stokes_numerical_test(${key}_poiseuille_gmres ${backend} poiseuille 1 1 1
      "upper triangular" apply apply 1 GMRES)
    # Positive diagonal Jacobi application gives MINRES an SPD preconditioner.
    stokes_numerical_test(${key}_poiseuille_minres ${backend} poiseuille 1 1 1
      diagonal apply apply 1 MINRES)

    stokes_numerical_test(${key}_backward_step ${backend} backward_step 1 0 1
      "lower triangular" solve solve 1 FGMRES)
    foreach(lid leaky watertight regularized)
      stokes_numerical_test(${key}_cavity_${lid} ${backend} cavity 1 1 1
        "upper triangular" solve solve 1 FGMRES LID ${lid})
    endforeach()
    stokes_numerical_test(${key}_refinement_cycles ${backend} poiseuille 1 0 0.7
      "upper triangular" solve solve 1 FGMRES CYCLES 2 CHECK_OUTPUT
      LABELS "refinement;output")
    if(DEAL_II_MPIEXEC)
      stokes_numerical_test(${key}_poiseuille_mpi2 ${backend} poiseuille 1 1 1
        "upper triangular" solve solve 2 FGMRES)
      stokes_numerical_test(${key}_poiseuille_gmres_mpi2 ${backend} poiseuille 1 1 1
        "upper triangular" apply apply 2 GMRES)
      stokes_numerical_test(${key}_poiseuille_minres_mpi2 ${backend} poiseuille 1 1 1
        diagonal apply apply 2 MINRES)
      stokes_numerical_test(${key}_colliding_mpi2_output ${backend} colliding 3 0 0.3
        "upper triangular" solve apply 2 FGMRES CHECK_OUTPUT
        LABELS "mpi;pressure-gauge;output")
    endif()
  endforeach()

  if(DEAL_II_WITH_PSBLAS AND DEAL_II_WITH_AMG4PSBLAS)
    set(ml_options
"      set MIN_COARSE_SIZE = 4
      set SMOOTHER_TYPE = FBGS
      set COARSE_SOLVE = BJAC
      set COARSE_SUBSOLVE = ILU
      set Additional options = hierarchy integer MAX_LEVS 6
      set Options file = options/velocity.options
      set Verbosity = true
      subsection PRE
        set SMOOTHER_SWEEPS = 1
      end
      subsection POST
        set SMOOTHER_SWEEPS = 1
      end")
    # The relative options-file path is resolved beside setup.prm. Its PRE
    # setting overrides the named entry after hierarchy construction.
    set(ml_file_options
"# Native smoother-phase settings, applied after the named and inline entries.
smoother integer SMOOTHER_SWEEPS 2 position=PRE
smoother integer SMOOTHER_SWEEPS 1 position=POST
")
    if(DEAL_II_MPIEXEC)
      set(ml_ranks 2)
    else()
      set(ml_ranks 1)
    endif()
    stokes_numerical_test(psblas_ml_native_options PSBLAS poiseuille 1 1 1
      "upper triangular" solve solve ${ml_ranks} FGMRES
      VELOCITY_PTYPE ML VELOCITY_NATIVE "${ml_options}"
      OPTIONS_FILE_CONTENT "${ml_file_options}" CHECK_MULTILEVEL
      LABELS "native-options;multilevel")
    if(DEAL_II_MPIEXEC)
      # Four coarse cells on five ranks force at least one empty cell owner.
      stokes_numerical_test(psblas_empty_rank PSBLAS poiseuille 1 0 1
        "upper triangular" solve solve 5 FGMRES
        LABELS "mpi;empty-ownership")
      # The old adapter rebuilt AMG on a reduced communicator. Exercise the
      # native multilevel hierarchy on the full communicator with an empty rank.
      stokes_numerical_test(psblas_empty_rank_ml PSBLAS poiseuille 1 0 1
        "upper triangular" solve solve 5 FGMRES
        VELOCITY_PTYPE ML VELOCITY_NATIVE "${ml_options}"
        OPTIONS_FILE_CONTENT "${ml_file_options}" CHECK_MULTILEVEL
        LABELS "mpi;empty-ownership;multilevel")
    endif()
  endif()

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
foreach(option "MAX_LEVS" "SUB_SOLVE" "pc_type" "Pressure degree" "Block preconditioner" "subsection LinearSolver")
  string(FIND "${contents}" "${option}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "Generated defaults are missing ${option}")
  endif()
endforeach()
if(NOT contents MATCHES "set[ \t]+Method[ \t]*=[ \t]*FGMRES[ \t\r]*\n")
  message(FATAL_ERROR "Generated defaults do not select Method = FGMRES")
endif()
if(contents MATCHES "subsection[ \t]+FGMRES[ \t\r]*\n")
  message(FATAL_ERROR "Generated defaults still contain the old FGMRES subsection")
endif()
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
