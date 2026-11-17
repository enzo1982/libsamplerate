option(LIBSAMPLERATE_COMPATIBLE_NAME "Use old dll name on Windows platform" OFF)

option(LIBSAMPLERATE_ENABLE_SINC_FAST_CONVERTER "Enable Fastest Sinc Interpolator converter" ON)
option(LIBSAMPLERATE_ENABLE_SINC_MEDIUM_CONVERTER "Enable Medium Sinc Interpolator converter" ON)
option(LIBSAMPLERATE_ENABLE_SINC_BEST_CONVERTER "Enable Best Sinc Interpolator converter" ON)

option(LIBSAMPLERATE_INSTALL_PKGCONFIG_MODULE "Install samplerate.pc PkgConfig module." ON)

set(ENABLE_SINC_FAST_CONVERTER ${LIBSAMPLERATE_ENABLE_SINC_FAST_CONVERTER} PARENT_SCOPE)
set(ENABLE_SINC_MEDIUM_CONVERTER ${LIBSAMPLERATE_ENABLE_SINC_MEDIUM_CONVERTER} PARENT_SCOPE)
set(ENABLE_SINC_BEST_CONVERTER ${LIBSAMPLERATE_ENABLE_SINC_BEST_CONVERTER} PARENT_SCOPE)

include(CMakePushCheckState)
include(CheckCSourceCompiles)
include(ClipMode)
include(CMakePackageConfigHelpers)

# This will set CPU_CLIPS_NEGATIVE and CPU_CLIPS_POSITIVE
clip_mode()

# check for symbol visibility attributes support
cmake_push_check_state()
set(CMAKE_REQUIRED_FLAGS "-fvisibility=hidden -Werror")
check_c_source_compiles("__attribute__((visibility(\"default\"))) int foo(void);
                         __attribute__((visibility(\"hidden\")))  int bar(void);
                         int foo (void) {return 0;}
                         int bar (void) {return 1;}
                         int main(void) {return 0;}" HAVE_VISIBILITY)
cmake_pop_check_state()

add_library(samplerate
  common.h
  fastest_coeffs.h
  high_qual_coeffs.h
  mid_qual_coeffs.h
  samplerate.c
  ${PROJECT_SOURCE_DIR}/include/samplerate.h
  src_linear.c
  src_sinc.c
  src_sinc.h
  src_sinc_avx2.c
  src_sinc_neon.c
  src_sinc_rvv.c
  src_sinc_sse2.c
  src_zoh.c
  $<$<AND:$<BOOL:${WIN32}>,$<BOOL:${BUILD_SHARED_LIBS}>>:../Win32/libsamplerate-0.def>)

# The SIMD versions of the inner loops of the sinc converters are built for their
# instruction sets using compiler flags for their files only. Which version gets
# used is decided at run time, so the library still works on CPUs without them.
# Each file compiles to an unused version if its instruction set is unavailable.

function(libsamplerate_simd_flags name file test_source)
  # Try no flags first, then the candidates passed after the test source.
  foreach(flags NONE ${ARGN})
    string(MAKE_C_IDENTIFIER "LIBSAMPLERATE_HAVE_${name}_${flags}" result)
    cmake_push_check_state(RESET)
    if(NOT flags STREQUAL "NONE")
      set(CMAKE_REQUIRED_FLAGS "${flags}")
    endif()
    set(CMAKE_REQUIRED_QUIET ON)
    check_c_source_compiles("${test_source}" ${result})
    cmake_pop_check_state()
    if(${result})
      if(flags STREQUAL "NONE")
        message(STATUS "Sinc converter ${name} version: enabled")
      else()
        set_property(SOURCE ${file} APPEND PROPERTY COMPILE_OPTIONS ${flags})
        message(STATUS "Sinc converter ${name} version: enabled (${flags})")
      endif()
      return()
    endif()
  endforeach()
  message(STATUS "Sinc converter ${name} version: not available")
endfunction()

libsamplerate_simd_flags(SSE2 src_sinc_sse2.c "
  #if !(defined (__SSE2__) || defined (_M_X64) || (defined (_M_IX86_FP) && _M_IX86_FP >= 2)) || defined (_M_ARM64EC)
  #error SSE2 not enabled
  #endif
  #include <emmintrin.h>
  int main (void) { __m128d a = _mm_set1_pd (1.0) ; return (int) _mm_cvtsd_f64 (_mm_add_pd (a, a)) ; }"
  -msse2 /arch:SSE2)

libsamplerate_simd_flags(AVX2 src_sinc_avx2.c "
  #if !defined (__AVX2__) || defined (_M_ARM64EC)
  #error AVX2 not enabled
  #endif
  #include <immintrin.h>
  int main (void) { __m256i a = _mm256_set1_epi32 (1) ; a = _mm256_add_epi64 (a, a) ; return _mm_cvtsi128_si32 (_mm256_castsi256_si128 (a)) ; }"
  -mavx2 /arch:AVX2)

libsamplerate_simd_flags(RVV src_sinc_rvv.c "
  #if !defined (__riscv_v) || !defined (__riscv_v_intrinsic) || __riscv_v_intrinsic < 12000
  #error RVV 1.0 intrinsics not available
  #endif
  #include <riscv_vector.h>
  int main (void) { size_t vl = __riscv_vsetvl_e32m1 (4) ; return (int) __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfmv_v_f_f32m1 (1.0f, vl)) ; }"
  -march=rv64gcv)

# ALIAS to use if libsamplerate is included from other project with add_subdirectory()
add_library(SampleRate::samplerate ALIAS samplerate)

# CMake generates wrong DLL names for MinGW and Cygwin, fix it
if(BUILD_SHARED_LIBS AND WIN32)
  if(LIBSAMPLERATE_COMPATIBLE_NAME)
    if(MSVC)
      set_target_properties(samplerate PROPERTIES OUTPUT_NAME "libsamplerate-${libsamplerate_VERSION_MAJOR}")
    else()
      set_target_properties(samplerate PROPERTIES OUTPUT_NAME "samplerate-${libsamplerate_VERSION_MAJOR}")
    endif()
  else()
    if(MINGW OR CYGWIN)
      set_target_properties(samplerate PROPERTIES RUNTIME_OUTPUT_NAME "samplerate-${libsamplerate_VERSION_MAJOR}")
    endif()
  endif()
  if(OS2)
    # OS/2 doesn't support a DLL name longer than 8 characters.
    set_target_properties(samplerate PROPERTIES OUTPUT_NAME "SAMPRATE")
    set_target_properties(samplerate PROPERTIES PREFIX "")
  endif()
endif()

target_include_directories(samplerate
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(LIBM_REQUIRED)
    target_link_libraries(samplerate PRIVATE m)
endif()

# Set public header
set_property(TARGET samplerate PROPERTY PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/include/samplerate.h)

if(BUILD_SHARED_LIBS)
  # Set ABI version. This is critical for Unix-like OSes
  set_target_properties(samplerate PROPERTIES
    VERSION ${libsamplerate_VERSION}
    SOVERSION ${libsamplerate_VERSION_MAJOR})

    # Use Version_script to export ABI set
    if(UNIX AND (NOT APPLE))
      if((CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
         (CMAKE_C_COMPILER_ID STREQUAL "Clang") OR
         (CMAKE_C_COMPILER_ID STREQUAL "Intel"))

        set(PACKAGE ${PROJECT_NAME})
        configure_file(Version_script.in Version_script)
        unset(PACKAGE)

        if(CMAKE_VERSION VERSION_LESS 3.13)
          # This works 
          set_property(TARGET samplerate APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/Version_script")
        else()
          # This works even better, e.g. for Clang it uses `-Xlinker` option,
          # but requires CMake >= 3.13.
          target_link_options(samplerate
            PRIVATE
              "LINKER:--version-script,${CMAKE_CURRENT_BINARY_DIR}/Version_script")
        endif()

      endif()
    endif()

  # keep in sync with configure.ac
  set (LIBSAMPLERATE_ABI_VERSION_CURRENT 2)
  set (LIBSAMPLERATE_ABI_VERSION_REVISION 2)
  set (LIBSAMPLERATE_ABI_VERSION_AGE 2)
  math (EXPR LIBSAMPLERATE_MACHO_COMPATIBILITY_VERSION "${LIBSAMPLERATE_ABI_VERSION_CURRENT} + 1")
  set (LIBSAMPLERATE_MACHO_CURRENT_VERSION "${LIBSAMPLERATE_MACHO_COMPATIBILITY_VERSION}.${LIBSAMPLERATE_ABI_VERSION_REVISION}.0")
  if (APPLE)
    if (NOT (CMAKE_VERSION VERSION_LESS 3.17))
      set_target_properties (samplerate PROPERTIES
        MACHO_CURRENT_VERSION ${LIBSAMPLERATE_MACHO_CURRENT_VERSION}
        MACHO_COMPATIBILITY_VERSION ${LIBSAMPLERATE_MACHO_COMPATIBILITY_VERSION}
        )
    else ()
      message (FATAL_ERROR "Apple platform requires cmake >= 3.17 to build dylib.")
    endif ()
  endif ()
endif()

# Istallation

if(LIBSAMPLERATE_INSTALL)

  install(TARGETS samplerate EXPORT SampleRateTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # pkg-config module

  if(LIBSAMPLERATE_INSTALL_PKGCONFIG_MODULE)
    set(prefix ${CMAKE_INSTALL_PREFIX})
    set(exec_prefix "\${prefix}")
    set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    set(libdir "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
    set(VERSION "${PROJECT_VERSION}")
    if(LIBM_REQUIRED)
      set(LIBS "-lm")
    endif()
    configure_file(../samplerate.pc.in samplerate.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/samplerate.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
  endif()

  set(CMAKE_INSTALL_PACKAGEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SampleRate")

  configure_package_config_file(../cmake/SampleRateConfig.cmake.in SampleRateConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_PACKAGEDIR})

  write_basic_package_version_file(SampleRateConfigVersion.cmake COMPATIBILITY SameMajorVersion)
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/SampleRateConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/SampleRateConfigVersion.cmake
    DESTINATION
      ${CMAKE_INSTALL_PACKAGEDIR})

  install(EXPORT SampleRateTargets
    NAMESPACE SampleRate::
    DESTINATION ${CMAKE_INSTALL_PACKAGEDIR})

endif()
