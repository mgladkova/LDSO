add_executable( run_test_superpoint test_SuperPoint.cc )
target_link_libraries( run_test_superpoint
  ldso ${THIRD_PARTY_LIBS})
set_property(TARGET run_test_superpoint PROPERTY CXX_STANDARD 14)