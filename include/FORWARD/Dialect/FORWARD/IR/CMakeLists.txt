add_mlir_dialect(FORWARDOps FORWARD)
add_mlir_doc(FORWARDOps FORWARDOps Dialects/ -gen-op-doc )

set(LLVM_TARGET_DEFINITIONS FORWARDOps.td)
mlir_tablegen(FORWARDOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(FORWARDOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRFORWARDOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS FORWARDOps.td)
mlir_tablegen(FORWARDOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(FORWARDOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRFORWARDOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS FORWARDOps.td)
mlir_tablegen(FORWARDOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=FORWARD)
mlir_tablegen(FORWARDOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=FORWARD)
add_public_tablegen_target(MLIRFORWARDOpsAttributesIncGen)