add_mlir_dialect_library(MLIRFORWARDMisc
    MiscPasses.cpp
    # AffineDataCopyGen.cpp
    # AffineLoopTiling.cpp
    # AffineLoopPermutation.cpp
    # MemrefEraseDealloc.cpp
    # ForwardMemrefAlloc.cpp
    # ForwardLinalgOps.cpp
    # FuncToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/

    DEPENDS
    MLIRFORWARDMiscPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
    MLIRPass
	)
