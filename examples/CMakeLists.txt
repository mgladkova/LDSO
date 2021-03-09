# TUM_MONO dataset
add_executable( run_dso_tum_mono run_dso_tum_mono.cc )
target_link_libraries( run_dso_tum_mono
  ldso ${THIRD_PARTY_LIBS})
set_property(TARGET run_dso_tum_mono PROPERTY CXX_STANDARD 14)

# EUROC dataset
add_executable( run_dso_euroc run_dso_euroc.cc )
target_link_libraries( run_dso_euroc
  ldso ${THIRD_PARTY_LIBS})
set_property(TARGET run_dso_euroc PROPERTY CXX_STANDARD 14)

# Kitti dataset
add_executable( run_dso_kitti run_dso_kitti.cc )
target_link_libraries( run_dso_kitti
  ldso ${THIRD_PARTY_LIBS})
set_property(TARGET run_dso_kitti PROPERTY CXX_STANDARD 14)