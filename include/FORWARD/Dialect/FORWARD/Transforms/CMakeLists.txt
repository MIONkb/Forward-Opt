add_mlir_doc(Passes FORWARDPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FORWARD)
add_public_tablegen_target(FORWARDPassIncGen)

