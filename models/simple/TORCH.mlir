module attributes {torch.debug_module_name = "simple"} {
  func.func @forward(%arg0: !torch.vtensor<[1,3,16,16],f32>) -> !torch.vtensor<[1,8,16,4],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %0 = torch.vtensor.literal(dense_resource<__elided__> : tensor<4x8xf32>) : !torch.vtensor<[4,8],f32>
    %1 = torch.vtensor.literal(dense_resource<__elided__> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %2 = torch.vtensor.literal(dense_resource<__elided__> : tensor<8x16xf32>) : !torch.vtensor<[8,16],f32>
    %3 = torch.vtensor.literal(dense_resource<__elided__> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %4 = torch.vtensor.literal(dense<0.000000e+00> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %5 = torch.vtensor.literal(dense<1.000000e+00> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %6 = torch.vtensor.literal(dense_resource<__elided__> : tensor<8x3x3x3xf32>) : !torch.vtensor<[8,3,3,3],f32>
    %7 = torch.vtensor.literal(dense_resource<__elided__> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %8 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %9 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %10 = torch.aten.convolution %arg0, %6, %7, %8, %8, %8, %false, %9, %int1 : !torch.vtensor<[1,3,16,16],f32>, !torch.vtensor<[8,3,3,3],f32>, !torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,8,16,16],f32>
    %11 = torch.aten.batch_norm %10, %5, %4, %4, %5, %false, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[1,8,16,16],f32>, !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[1,8,16,16],f32>
    %12 = torch.aten.relu %11 : !torch.vtensor<[1,8,16,16],f32> -> !torch.vtensor<[1,8,16,16],f32>
    %13 = torch.aten.transpose.int %2, %int0, %int1 : !torch.vtensor<[8,16],f32>, !torch.int, !torch.int -> !torch.vtensor<[16,8],f32>
    %14 = torch.aten.matmul %12, %13 : !torch.vtensor<[1,8,16,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[1,8,16,8],f32>
    %15 = torch.aten.add.Tensor %14, %3, %float1.000000e00 : !torch.vtensor<[1,8,16,8],f32>, !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[1,8,16,8],f32>
    %16 = torch.aten.relu %15 : !torch.vtensor<[1,8,16,8],f32> -> !torch.vtensor<[1,8,16,8],f32>
    %17 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[4,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[8,4],f32>
    %18 = torch.aten.matmul %16, %17 : !torch.vtensor<[1,8,16,8],f32>, !torch.vtensor<[8,4],f32> -> !torch.vtensor<[1,8,16,4],f32>
    %19 = torch.aten.add.Tensor %18, %1, %float1.000000e00 : !torch.vtensor<[1,8,16,4],f32>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[1,8,16,4],f32>
    return %19 : !torch.vtensor<[1,8,16,4],f32>
  }
}

