cmake_minimum_required(VERSION 3.16)
project(TRIAD_REAL)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenFHE REQUIRED)
include_directories(${OpenFHE_INCLUDE_DIRS})
link_directories(${OpenFHE_LIBRARY_DIRS})

if(WIN32)
  include_directories(D:/CKKS/openfhe-install/include)
  include_directories(D:/CKKS/openfhe-install/include/openfhe)
  include_directories(D:/CKKS/openfhe-install/include/openfhe/core)
  include_directories(D:/CKKS/openfhe-install/include/openfhe/pke)
  include_directories(D:/CKKS/openfhe-install/include/openfhe/binfhe)
  link_directories(D:/CKKS/openfhe-install/lib)
else()
  include_directories(/usr/local/include/openfhe)
  include_directories(/usr/local/include/openfhe/core)
  include_directories(/usr/local/include/openfhe/pke)
  include_directories(/usr/local/include/openfhe/binfhe)
  link_directories(/usr/local/lib)
endif()

# --- keygen_real: generates keys with BatchSize=8192 for N_FEAT=4088 ---
add_executable(keygen_real keygen_real.cpp)

# --- server_real: runs on Windows laptop ---
add_executable(server_real server_real.cpp)

# --- client_real: runs on Raspberry Pis ---
add_executable(client_real client_real.cpp)

if(WIN32)
    # MinGW COFF section limit fix
    foreach(tgt keygen_real server_real client_real)
        target_compile_options(${tgt} PRIVATE -Wa,-mbig-obj)
    endforeach()
    target_link_libraries(keygen_real OPENFHEpke)
    target_link_libraries(server_real OPENFHEpke ws2_32)
    target_link_libraries(client_real OPENFHEpke ws2_32)
else()
    target_link_libraries(keygen_real OPENFHEpke OPENFHEcore OPENFHEbinfhe)
    target_link_libraries(server_real OPENFHEpke OPENFHEcore OPENFHEbinfhe)
    target_link_libraries(client_real OPENFHEpke OPENFHEcore OPENFHEbinfhe)
    foreach(tgt keygen_real server_real client_real)
        target_link_options(${tgt} PRIVATE
            -Wl,--whole-archive
            /usr/local/lib/libOPENFHEpke.so
            -Wl,--no-whole-archive)
    endforeach()
endif()
