#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "simple"} {
  func.func @forward(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x8x16x4xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<8x3x3x3xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x16x8xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1x8x4xf32>
    %cst_2 = arith.constant dense<1.000010e+00> : tensor<8x1x1xf32>
    %cst_3 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
    %cst_4 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<8xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<8xf32>
    %cst_7 = arith.constant dense<[-0.121429406, 0.0909240693, 0.0867559984, 0.169947132]> : tensor<4xf32>
    %0 = "tosa.transpose"(%arg0, %cst_4) : (tensor<1x3x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x3xf32>
    %cst_8 = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_8 : f32
    } : tensor<1x16x16x3xf32> to tensor<1x18x18x3xf32>
    %cst_9 = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi64>
    %1 = "tosa.transpose"(%cst, %cst_9) : (tensor<8x3x3x3xf32>, tensor<4xi64>) -> tensor<3x3x3x8xf32>
    %2 = tensor.empty() : tensor<1x16x16x8xf32>
    %cst_10 = arith.constant 0.000000e+00 : f32
    %3 = linalg.fill ins(%cst_10 : f32) outs(%2 : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
    %4 = tensor.empty() : tensor<1x16x16x8xf32>
    %5 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %1 : tensor<1x18x18x3xf32>, tensor<3x3x3x8xf32>) outs(%3 : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5, %5 : tensor<8xf32>, tensor<1x16x16x8xf32>) outs(%4 : tensor<1x16x16x8xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %20 = arith.addf %in, %in_15 : f32
      linalg.yield %20 : f32
    } -> tensor<1x16x16x8xf32>
    %7 = "tosa.transpose"(%6, %cst_3) : (tensor<1x16x16x8xf32>, tensor<4xi32>) -> tensor<1x8x16x16xf32>
    %8 = "tosa.rsqrt"(%cst_2) : (tensor<8x1x1xf32>) -> tensor<8x1x1xf32>
    %9 = "tosa.mul"(%7, %8) <{shift = 0 : i32}> : (tensor<1x8x16x16xf32>, tensor<8x1x1xf32>) -> tensor<1x8x16x16xf32>
    %10 = "tosa.clamp"(%9) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %collapsed = tensor.collapse_shape %10 [[0], [1, 2], [3]] : tensor<1x8x16x16xf32> into tensor<1x128x16xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %11 = tensor.empty() : tensor<1x128x8xf32>
    %12 = linalg.fill ins(%cst_11 : f32) outs(%11 : tensor<1x128x8xf32>) -> tensor<1x128x8xf32>
    %13 = linalg.batch_matmul ins(%collapsed, %cst_0 : tensor<1x128x16xf32>, tensor<1x16x8xf32>) outs(%12 : tensor<1x128x8xf32>) -> tensor<1x128x8xf32>
    %expanded = tensor.expand_shape %13 [[0], [1, 2], [3]] : tensor<1x128x8xf32> into tensor<1x8x16x8xf32>
    %14 = "tosa.add"(%expanded, %cst_6) : (tensor<1x8x16x8xf32>, tensor<8xf32>) -> tensor<1x8x16x8xf32>
    %15 = "tosa.clamp"(%14) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x8x16x8xf32>) -> tensor<1x8x16x8xf32>
    %collapsed_12 = tensor.collapse_shape %15 [[0], [1, 2], [3]] : tensor<1x8x16x8xf32> into tensor<1x128x8xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %16 = tensor.empty() : tensor<1x128x4xf32>
    %17 = linalg.fill ins(%cst_13 : f32) outs(%16 : tensor<1x128x4xf32>) -> tensor<1x128x4xf32>
    %18 = linalg.batch_matmul ins(%collapsed_12, %cst_1 : tensor<1x128x8xf32>, tensor<1x8x4xf32>) outs(%17 : tensor<1x128x4xf32>) -> tensor<1x128x4xf32>
    %expanded_14 = tensor.expand_shape %18 [[0], [1, 2], [3]] : tensor<1x128x4xf32> into tensor<1x8x16x4xf32>
    %19 = "tosa.add"(%expanded_14, %cst_7) : (tensor<1x8x16x4xf32>, tensor<4xf32>) -> tensor<1x8x16x4xf32>
    return %19 : tensor<1x8x16x4xf32>
  }
}

