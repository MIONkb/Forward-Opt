module attributes {torch.debug_module_name = "simple"} {
  func.func @forward(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x8x16x4xf32> {
    %0 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<8x3x3x3xf32>}> {idx = "0", npz_loc = "0.tosa.const"} : () -> tensor<8x3x3x3xf32>
    %1 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1x16x8xf32>}> {idx = "1", npz_loc = "1.tosa.const"} : () -> tensor<1x16x8xf32>
    %2 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1x8x4xf32>}> {idx = "2", npz_loc = "2.tosa.const"} : () -> tensor<1x8x4xf32>
    %3 = "tosa.const"() <{value = dense<1.000010e+00> : tensor<8x1x1xf32>}> {idx = "3", npz_loc = "3.tosa.const"} : () -> tensor<8x1x1xf32>
    %4 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> {idx = "4", npz_loc = "4.tosa.const"} : () -> tensor<4xi32>
    %5 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> {idx = "5", npz_loc = "5.tosa.const"} : () -> tensor<4xi32>
    %6 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<8xf32>}> {idx = "6", npz_loc = "6.tosa.const"} : () -> tensor<8xf32>
    %7 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<8xf32>}> {idx = "7", npz_loc = "7.tosa.const"} : () -> tensor<8xf32>
    %8 = "tosa.const"() <{value = dense<[-0.121429406, 0.0909240693, 0.0867559984, 0.169947132]> : tensor<4xf32>}> {idx = "8", npz_loc = "8.tosa.const"} : () -> tensor<4xf32>
    %9 = "tosa.transpose"(%arg0, %5) {idx = "9"} : (tensor<1x3x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x3xf32>
    %10 = "tosa.conv2d"(%9, %0, %6) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> {idx = "10"} : (tensor<1x16x16x3xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x16x16x8xf32>
    %11 = "tosa.transpose"(%10, %4) {idx = "11"} : (tensor<1x16x16x8xf32>, tensor<4xi32>) -> tensor<1x8x16x16xf32>
    %12 = "tosa.rsqrt"(%3) {idx = "12"} : (tensor<8x1x1xf32>) -> tensor<8x1x1xf32>
    %13 = "tosa.mul"(%11, %12) <{shift = 0 : i32}> {idx = "13"} : (tensor<1x8x16x16xf32>, tensor<8x1x1xf32>) -> tensor<1x8x16x16xf32>
    %14 = "tosa.clamp"(%13) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> {idx = "14"} : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %15 = "tosa.reshape"(%14) <{new_shape = array<i64: 1, 128, 16>}> {idx = "15"} : (tensor<1x8x16x16xf32>) -> tensor<1x128x16xf32>
    %16 = "tosa.matmul"(%15, %1) {idx = "16"} : (tensor<1x128x16xf32>, tensor<1x16x8xf32>) -> tensor<1x128x8xf32>
    %17 = "tosa.reshape"(%16) <{new_shape = array<i64: 1, 8, 16, 8>}> {idx = "17"} : (tensor<1x128x8xf32>) -> tensor<1x8x16x8xf32>
    %18 = "tosa.add"(%17, %7) {idx = "18"} : (tensor<1x8x16x8xf32>, tensor<8xf32>) -> tensor<1x8x16x8xf32>
    %19 = "tosa.clamp"(%18) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> {idx = "19"} : (tensor<1x8x16x8xf32>) -> tensor<1x8x16x8xf32>
    %20 = "tosa.reshape"(%19) <{new_shape = array<i64: 1, 128, 8>}> {idx = "20"} : (tensor<1x8x16x8xf32>) -> tensor<1x128x8xf32>
    %21 = "tosa.matmul"(%20, %2) {idx = "21"} : (tensor<1x128x8xf32>, tensor<1x8x4xf32>) -> tensor<1x128x4xf32>
    %22 = "tosa.reshape"(%21) <{new_shape = array<i64: 1, 8, 16, 4>}> {idx = "22"} : (tensor<1x128x4xf32>) -> tensor<1x8x16x4xf32>
    %23 = "tosa.add"(%22, %8) {idx = "23"} : (tensor<1x8x16x4xf32>, tensor<4xf32>) -> tensor<1x8x16x4xf32>
    return %23 : tensor<1x8x16x4xf32>
  }
}

