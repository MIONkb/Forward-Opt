#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map15 = affine_map<(d0, d1, d2, d3) -> ()>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map17 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
"builtin.module"() ({
  "ml_program.global"() {is_mutable, sym_name = "global_seed", sym_visibility = "private", type = tensor<i64>, value = dense<0> : tensor<i64>} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<1x197x768xf32>):
    %0 = "arith.constant"() {value = 0 : i64} : () -> i64
    %1 = "arith.constant"() : () -> tensor<1000x768xf32>
    %2 = "arith.constant"() : () -> tensor<1000xf32>
    %3 = "arith.constant"() : () -> tensor<768xf32>
    %4 = "arith.constant"() : () -> tensor<768xf32>
    %5 = "arith.constant"() : () -> tensor<768x3072xf32>
    %6 = "arith.constant"() : () -> tensor<768xf32>
    %7 = "arith.constant"() : () -> tensor<3072x768xf32>
    %8 = "arith.constant"() : () -> tensor<3072xf32>
    %9 = "arith.constant"() : () -> tensor<768xf32>
    %10 = "arith.constant"() : () -> tensor<768xf32>
    %11 = "arith.constant"() : () -> tensor<768x768xf32>
    %12 = "arith.constant"() : () -> tensor<768xf32>
    %13 = "arith.constant"() : () -> tensor<768x768xf32>
    %14 = "arith.constant"() : () -> tensor<768xf32>
    %15 = "arith.constant"() : () -> tensor<768x768xf32>
    %16 = "arith.constant"() : () -> tensor<768xf32>
    %17 = "arith.constant"() : () -> tensor<768x768xf32>
    %18 = "arith.constant"() : () -> tensor<768xf32>
    %19 = "arith.constant"() : () -> tensor<768xf32>
    %20 = "arith.constant"() : () -> tensor<768xf32>
    %21 = "arith.constant"() : () -> tensor<768x3072xf32>
    %22 = "arith.constant"() : () -> tensor<768xf32>
    %23 = "arith.constant"() : () -> tensor<3072x768xf32>
    %24 = "arith.constant"() : () -> tensor<3072xf32>
    %25 = "arith.constant"() : () -> tensor<768xf32>
    %26 = "arith.constant"() : () -> tensor<768xf32>
    %27 = "arith.constant"() : () -> tensor<768x768xf32>
    %28 = "arith.constant"() : () -> tensor<768xf32>
    %29 = "arith.constant"() : () -> tensor<768x768xf32>
    %30 = "arith.constant"() : () -> tensor<768xf32>
    %31 = "arith.constant"() : () -> tensor<768x768xf32>
    %32 = "arith.constant"() : () -> tensor<768xf32>
    %33 = "arith.constant"() : () -> tensor<768x768xf32>
    %34 = "arith.constant"() : () -> tensor<768xf32>
    %35 = "arith.constant"() : () -> tensor<768xf32>
    %36 = "arith.constant"() : () -> tensor<768xf32>
    %37 = "arith.constant"() : () -> tensor<768x3072xf32>
    %38 = "arith.constant"() : () -> tensor<768xf32>
    %39 = "arith.constant"() : () -> tensor<3072x768xf32>
    %40 = "arith.constant"() : () -> tensor<3072xf32>
    %41 = "arith.constant"() : () -> tensor<768xf32>
    %42 = "arith.constant"() : () -> tensor<768xf32>
    %43 = "arith.constant"() : () -> tensor<768x768xf32>
    %44 = "arith.constant"() : () -> tensor<768xf32>
    %45 = "arith.constant"() : () -> tensor<768x768xf32>
    %46 = "arith.constant"() : () -> tensor<768xf32>
    %47 = "arith.constant"() : () -> tensor<768x768xf32>
    %48 = "arith.constant"() : () -> tensor<768xf32>
    %49 = "arith.constant"() : () -> tensor<768x768xf32>
    %50 = "arith.constant"() : () -> tensor<768xf32>
    %51 = "arith.constant"() : () -> tensor<768xf32>
    %52 = "arith.constant"() : () -> tensor<768xf32>
    %53 = "arith.constant"() : () -> tensor<768x3072xf32>
    %54 = "arith.constant"() : () -> tensor<768xf32>
    %55 = "arith.constant"() : () -> tensor<3072x768xf32>
    %56 = "arith.constant"() : () -> tensor<3072xf32>
    %57 = "arith.constant"() : () -> tensor<768xf32>
    %58 = "arith.constant"() : () -> tensor<768xf32>
    %59 = "arith.constant"() : () -> tensor<768x768xf32>
    %60 = "arith.constant"() : () -> tensor<768xf32>
    %61 = "arith.constant"() : () -> tensor<768x768xf32>
    %62 = "arith.constant"() : () -> tensor<768xf32>
    %63 = "arith.constant"() : () -> tensor<768x768xf32>
    %64 = "arith.constant"() : () -> tensor<768xf32>
    %65 = "arith.constant"() : () -> tensor<768x768xf32>
    %66 = "arith.constant"() : () -> tensor<768xf32>
    %67 = "arith.constant"() : () -> tensor<768xf32>
    %68 = "arith.constant"() : () -> tensor<768xf32>
    %69 = "arith.constant"() : () -> tensor<768x3072xf32>
    %70 = "arith.constant"() : () -> tensor<768xf32>
    %71 = "arith.constant"() : () -> tensor<3072x768xf32>
    %72 = "arith.constant"() : () -> tensor<3072xf32>
    %73 = "arith.constant"() : () -> tensor<768xf32>
    %74 = "arith.constant"() : () -> tensor<768xf32>
    %75 = "arith.constant"() : () -> tensor<768x768xf32>
    %76 = "arith.constant"() : () -> tensor<768xf32>
    %77 = "arith.constant"() : () -> tensor<768x768xf32>
    %78 = "arith.constant"() : () -> tensor<768xf32>
    %79 = "arith.constant"() : () -> tensor<768x768xf32>
    %80 = "arith.constant"() : () -> tensor<768xf32>
    %81 = "arith.constant"() : () -> tensor<768x768xf32>
    %82 = "arith.constant"() : () -> tensor<768xf32>
    %83 = "arith.constant"() : () -> tensor<768xf32>
    %84 = "arith.constant"() : () -> tensor<768xf32>
    %85 = "arith.constant"() : () -> tensor<768x3072xf32>
    %86 = "arith.constant"() : () -> tensor<768xf32>
    %87 = "arith.constant"() : () -> tensor<3072x768xf32>
    %88 = "arith.constant"() : () -> tensor<3072xf32>
    %89 = "arith.constant"() : () -> tensor<768xf32>
    %90 = "arith.constant"() : () -> tensor<768xf32>
    %91 = "arith.constant"() : () -> tensor<768x768xf32>
    %92 = "arith.constant"() : () -> tensor<768xf32>
    %93 = "arith.constant"() : () -> tensor<768x768xf32>
    %94 = "arith.constant"() : () -> tensor<768xf32>
    %95 = "arith.constant"() : () -> tensor<768x768xf32>
    %96 = "arith.constant"() : () -> tensor<768xf32>
    %97 = "arith.constant"() : () -> tensor<768x768xf32>
    %98 = "arith.constant"() : () -> tensor<768xf32>
    %99 = "arith.constant"() : () -> tensor<768xf32>
    %100 = "arith.constant"() : () -> tensor<768xf32>
    %101 = "arith.constant"() : () -> tensor<768x3072xf32>
    %102 = "arith.constant"() : () -> tensor<768xf32>
    %103 = "arith.constant"() : () -> tensor<3072x768xf32>
    %104 = "arith.constant"() : () -> tensor<3072xf32>
    %105 = "arith.constant"() : () -> tensor<768xf32>
    %106 = "arith.constant"() : () -> tensor<768xf32>
    %107 = "arith.constant"() : () -> tensor<768x768xf32>
    %108 = "arith.constant"() : () -> tensor<768xf32>
    %109 = "arith.constant"() : () -> tensor<768x768xf32>
    %110 = "arith.constant"() : () -> tensor<768xf32>
    %111 = "arith.constant"() : () -> tensor<768x768xf32>
    %112 = "arith.constant"() : () -> tensor<768xf32>
    %113 = "arith.constant"() : () -> tensor<768x768xf32>
    %114 = "arith.constant"() : () -> tensor<768xf32>
    %115 = "arith.constant"() : () -> tensor<768xf32>
    %116 = "arith.constant"() : () -> tensor<768xf32>
    %117 = "arith.constant"() : () -> tensor<768x3072xf32>
    %118 = "arith.constant"() : () -> tensor<768xf32>
    %119 = "arith.constant"() : () -> tensor<3072x768xf32>
    %120 = "arith.constant"() : () -> tensor<3072xf32>
    %121 = "arith.constant"() : () -> tensor<768xf32>
    %122 = "arith.constant"() : () -> tensor<768xf32>
    %123 = "arith.constant"() : () -> tensor<768x768xf32>
    %124 = "arith.constant"() : () -> tensor<768xf32>
    %125 = "arith.constant"() : () -> tensor<768x768xf32>
    %126 = "arith.constant"() : () -> tensor<768xf32>
    %127 = "arith.constant"() : () -> tensor<768x768xf32>
    %128 = "arith.constant"() : () -> tensor<768xf32>
    %129 = "arith.constant"() : () -> tensor<768x768xf32>
    %130 = "arith.constant"() : () -> tensor<768xf32>
    %131 = "arith.constant"() : () -> tensor<768xf32>
    %132 = "arith.constant"() : () -> tensor<768xf32>
    %133 = "arith.constant"() : () -> tensor<768x3072xf32>
    %134 = "arith.constant"() : () -> tensor<768xf32>
    %135 = "arith.constant"() : () -> tensor<3072x768xf32>
    %136 = "arith.constant"() : () -> tensor<3072xf32>
    %137 = "arith.constant"() : () -> tensor<768xf32>
    %138 = "arith.constant"() : () -> tensor<768xf32>
    %139 = "arith.constant"() : () -> tensor<768x768xf32>
    %140 = "arith.constant"() : () -> tensor<768xf32>
    %141 = "arith.constant"() : () -> tensor<768x768xf32>
    %142 = "arith.constant"() : () -> tensor<768xf32>
    %143 = "arith.constant"() : () -> tensor<768x768xf32>
    %144 = "arith.constant"() : () -> tensor<768xf32>
    %145 = "arith.constant"() : () -> tensor<768x768xf32>
    %146 = "arith.constant"() : () -> tensor<768xf32>
    %147 = "arith.constant"() : () -> tensor<768xf32>
    %148 = "arith.constant"() : () -> tensor<768xf32>
    %149 = "arith.constant"() : () -> tensor<768x3072xf32>
    %150 = "arith.constant"() : () -> tensor<768xf32>
    %151 = "arith.constant"() : () -> tensor<3072x768xf32>
    %152 = "arith.constant"() : () -> tensor<3072xf32>
    %153 = "arith.constant"() : () -> tensor<768xf32>
    %154 = "arith.constant"() : () -> tensor<768xf32>
    %155 = "arith.constant"() : () -> tensor<768x768xf32>
    %156 = "arith.constant"() : () -> tensor<768xf32>
    %157 = "arith.constant"() : () -> tensor<768x768xf32>
    %158 = "arith.constant"() : () -> tensor<768xf32>
    %159 = "arith.constant"() : () -> tensor<768x768xf32>
    %160 = "arith.constant"() : () -> tensor<768xf32>
    %161 = "arith.constant"() : () -> tensor<768x768xf32>
    %162 = "arith.constant"() : () -> tensor<768xf32>
    %163 = "arith.constant"() : () -> tensor<768xf32>
    %164 = "arith.constant"() : () -> tensor<768xf32>
    %165 = "arith.constant"() : () -> tensor<768x3072xf32>
    %166 = "arith.constant"() : () -> tensor<768xf32>
    %167 = "arith.constant"() : () -> tensor<3072x768xf32>
    %168 = "arith.constant"() : () -> tensor<3072xf32>
    %169 = "arith.constant"() : () -> tensor<768xf32>
    %170 = "arith.constant"() : () -> tensor<768xf32>
    %171 = "arith.constant"() : () -> tensor<768x768xf32>
    %172 = "arith.constant"() : () -> tensor<768xf32>
    %173 = "arith.constant"() : () -> tensor<768x768xf32>
    %174 = "arith.constant"() : () -> tensor<768xf32>
    %175 = "arith.constant"() : () -> tensor<768x768xf32>
    %176 = "arith.constant"() : () -> tensor<768xf32>
    %177 = "arith.constant"() : () -> tensor<768x768xf32>
    %178 = "arith.constant"() : () -> tensor<768xf32>
    %179 = "arith.constant"() : () -> tensor<768xf32>
    %180 = "arith.constant"() : () -> tensor<768xf32>
    %181 = "arith.constant"() : () -> tensor<768x3072xf32>
    %182 = "arith.constant"() : () -> tensor<768xf32>
    %183 = "arith.constant"() : () -> tensor<3072x768xf32>
    %184 = "arith.constant"() : () -> tensor<3072xf32>
    %185 = "arith.constant"() : () -> tensor<768xf32>
    %186 = "arith.constant"() : () -> tensor<768xf32>
    %187 = "arith.constant"() : () -> tensor<768x768xf32>
    %188 = "arith.constant"() : () -> tensor<768xf32>
    %189 = "arith.constant"() : () -> tensor<f64>
    %190 = "arith.constant"() : () -> tensor<768x768xf32>
    %191 = "arith.constant"() : () -> tensor<768xf32>
    %192 = "arith.constant"() : () -> tensor<768x768xf32>
    %193 = "arith.constant"() : () -> tensor<768xf32>
    %194 = "arith.constant"() : () -> tensor<768x768xf32>
    %195 = "arith.constant"() : () -> tensor<768xf32>
    %196 = "arith.constant"() : () -> tensor<768xf32>
    %197 = "arith.constant"() : () -> tensor<768xf32>
    %198 = "arith.constant"() : () -> f32
    %199 = "arith.constant"() : () -> f32
    %200 = "arith.constant"() : () -> f32
    %201 = "arith.constant"() : () -> f32
    %202 = "arith.constant"() : () -> f32
    %203 = "arith.constant"() : () -> f64
    %204 = "arith.constant"() : () -> f32
    %205 = "tensor.empty"() : () -> tensor<1x197x1xf32>
    %206 = "linalg.fill"(%198, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %207 = "linalg.generic"(%arg0, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %208 = "linalg.generic"(%207, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %209 = "tensor.empty"() : () -> tensor<1x197x768xf32>
    %210 = "linalg.generic"(%208, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %211 = "linalg.generic"(%arg0, %210, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %212 = "linalg.generic"(%211, %211, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %213 = "linalg.generic"(%212, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %214 = "linalg.generic"(%213, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %215 = "linalg.generic"(%214, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %216 = "linalg.generic"(%215, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %217 = "linalg.generic"(%216, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %218 = "linalg.generic"(%211, %217, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %219 = "linalg.generic"(%218, %196, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %220 = "linalg.generic"(%219, %197, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %221 = "tensor.empty"() : () -> tensor<768x768xf32>
    %222 = "linalg.generic"(%194, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %223 = "linalg.generic"(%220, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %224 = "tensor.empty"() : () -> tensor<1x768x768xf32>
    %225 = "linalg.generic"(%222, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %226 = "linalg.fill"(%198, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %227 = "linalg.batch_matmul"(%223, %225, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %228 = "linalg.generic"(%227, %195, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %229 = "linalg.generic"(%192, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %230 = "linalg.generic"(%229, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %231 = "linalg.batch_matmul"(%223, %230, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %232 = "linalg.generic"(%231, %193, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %233 = "tensor.expand_shape"(%232) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %234 = "tensor.empty"() : () -> tensor<1x12x197x64xf32>
    %235 = "linalg.generic"(%233, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %236 = "linalg.generic"(%190, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %237 = "linalg.generic"(%236, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %238 = "linalg.batch_matmul"(%223, %237, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %239 = "linalg.generic"(%238, %191, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %240 = "tensor.expand_shape"(%239) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %241 = "linalg.generic"(%240, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %242 = "tensor.expand_shape"(%228) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %243 = "linalg.generic"(%242, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %244 = "tensor.empty"() : () -> tensor<1x12x64x197xf32>
    %245 = "linalg.generic"(%235, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %246 = "linalg.generic"(%243, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %247 = "linalg.generic"(%245, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %248 = "tensor.collapse_shape"(%246) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %249 = "tensor.collapse_shape"(%247) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %250 = "tensor.empty"() : () -> tensor<12x197x197xf32>
    %251 = "linalg.fill"(%198, %250) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %252 = "linalg.batch_matmul"(%248, %249, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %253 = "tensor.expand_shape"(%252) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %254 = "tensor.empty"() : () -> tensor<1x12x197x197xf32>
    %255 = "linalg.generic"(%253, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %256 = "tensor.empty"() : () -> tensor<1x12x197x1xi64>
    %257 = "linalg.fill"(%0, %256) ({
    ^bb0(%arg1: i64, %arg2: i64):
      "linalg.yield"(%arg1) : (i64) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (i64, tensor<1x12x197x1xi64>) -> tensor<1x12x197x1xi64>
    %258 = "tensor.empty"() : () -> tensor<1x12x197x1xf32>
    %259 = "linalg.fill"(%199, %258) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %260:2 = "linalg.generic"(%255, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %261 = "linalg.generic"(%255, %260#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %262 = "linalg.generic"(%261, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %263 = "linalg.fill"(%198, %258) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %264 = "linalg.generic"(%262, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %265 = "linalg.generic"(%262, %264, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %266 = "linalg.generic"(%265, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %267 = "linalg.generic"(%241, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %268 = "tensor.collapse_shape"(%266) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %269 = "tensor.collapse_shape"(%267) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %270 = "tensor.empty"() : () -> tensor<12x197x64xf32>
    %271 = "linalg.fill"(%198, %270) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %272 = "linalg.batch_matmul"(%268, %269, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %273 = "tensor.expand_shape"(%272) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %274 = "tensor.empty"() : () -> tensor<1x197x12x64xf32>
    %275 = "linalg.generic"(%273, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %276 = "tensor.collapse_shape"(%275) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %277 = "linalg.generic"(%187, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %278 = "linalg.generic"(%276, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %279 = "linalg.generic"(%277, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %280 = "linalg.batch_matmul"(%278, %279, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %281 = "linalg.generic"(%280, %188, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %282 = "linalg.generic"(%281, %arg0, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %283 = "linalg.generic"(%282, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %284 = "linalg.generic"(%283, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %285 = "linalg.generic"(%284, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %286 = "linalg.generic"(%282, %285, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %287 = "linalg.generic"(%286, %286, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %288 = "linalg.generic"(%287, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %289 = "linalg.generic"(%288, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %290 = "linalg.generic"(%289, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %291 = "linalg.generic"(%290, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %292 = "linalg.generic"(%291, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %293 = "linalg.generic"(%286, %292, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %294 = "linalg.generic"(%293, %185, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %295 = "linalg.generic"(%294, %186, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %296 = "tensor.empty"() : () -> tensor<768x3072xf32>
    %297 = "linalg.generic"(%183, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %298 = "linalg.generic"(%295, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %299 = "tensor.empty"() : () -> tensor<1x768x3072xf32>
    %300 = "linalg.generic"(%297, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %301 = "tensor.empty"() : () -> tensor<1x197x3072xf32>
    %302 = "linalg.fill"(%198, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %303 = "linalg.batch_matmul"(%298, %300, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %304 = "linalg.generic"(%303, %184, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %305 = "linalg.generic"(%304, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %306 = "tensor.empty"() : () -> tensor<3072x768xf32>
    %307 = "linalg.generic"(%181, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %308 = "linalg.generic"(%305, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %309 = "tensor.empty"() : () -> tensor<1x3072x768xf32>
    %310 = "linalg.generic"(%307, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %311 = "linalg.batch_matmul"(%308, %310, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %312 = "linalg.generic"(%311, %182, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %313 = "linalg.generic"(%312, %282, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %314 = "linalg.generic"(%313, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %315 = "linalg.generic"(%314, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %316 = "linalg.generic"(%315, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %317 = "linalg.generic"(%313, %316, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %318 = "linalg.generic"(%317, %317, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %319 = "linalg.generic"(%318, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %320 = "linalg.generic"(%319, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %321 = "linalg.generic"(%320, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %322 = "linalg.generic"(%321, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %323 = "linalg.generic"(%322, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %324 = "linalg.generic"(%317, %323, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %325 = "linalg.generic"(%324, %179, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %326 = "linalg.generic"(%325, %180, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %327 = "linalg.generic"(%177, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %328 = "linalg.generic"(%326, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %329 = "linalg.generic"(%327, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %330 = "linalg.batch_matmul"(%328, %329, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %331 = "linalg.generic"(%330, %178, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %332 = "linalg.generic"(%175, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %333 = "linalg.generic"(%332, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %334 = "linalg.batch_matmul"(%328, %333, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %335 = "linalg.generic"(%334, %176, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %336 = "tensor.expand_shape"(%335) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %337 = "linalg.generic"(%336, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %338 = "linalg.generic"(%173, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %339 = "linalg.generic"(%338, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %340 = "linalg.batch_matmul"(%328, %339, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %341 = "linalg.generic"(%340, %174, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %342 = "tensor.expand_shape"(%341) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %343 = "linalg.generic"(%342, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %344 = "tensor.expand_shape"(%331) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %345 = "linalg.generic"(%344, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %346 = "linalg.generic"(%337, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %347 = "linalg.generic"(%345, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %348 = "linalg.generic"(%346, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %349 = "tensor.collapse_shape"(%347) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %350 = "tensor.collapse_shape"(%348) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %351 = "linalg.batch_matmul"(%349, %350, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %352 = "tensor.expand_shape"(%351) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %353 = "linalg.generic"(%352, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %354:2 = "linalg.generic"(%353, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %355 = "linalg.generic"(%353, %354#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %356 = "linalg.generic"(%355, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %357 = "linalg.generic"(%356, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %358 = "linalg.generic"(%356, %357, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %359 = "linalg.generic"(%358, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %360 = "linalg.generic"(%343, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %361 = "tensor.collapse_shape"(%359) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %362 = "tensor.collapse_shape"(%360) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %363 = "linalg.batch_matmul"(%361, %362, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %364 = "tensor.expand_shape"(%363) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %365 = "linalg.generic"(%364, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %366 = "tensor.collapse_shape"(%365) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %367 = "linalg.generic"(%171, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %368 = "linalg.generic"(%366, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %369 = "linalg.generic"(%367, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %370 = "linalg.batch_matmul"(%368, %369, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %371 = "linalg.generic"(%370, %172, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %372 = "linalg.generic"(%371, %313, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %373 = "linalg.generic"(%372, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %374 = "linalg.generic"(%373, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %375 = "linalg.generic"(%374, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %376 = "linalg.generic"(%372, %375, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %377 = "linalg.generic"(%376, %376, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %378 = "linalg.generic"(%377, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %379 = "linalg.generic"(%378, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %380 = "linalg.generic"(%379, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %381 = "linalg.generic"(%380, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %382 = "linalg.generic"(%381, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %383 = "linalg.generic"(%376, %382, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %384 = "linalg.generic"(%383, %169, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %385 = "linalg.generic"(%384, %170, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %386 = "linalg.generic"(%167, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %387 = "linalg.generic"(%385, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %388 = "linalg.generic"(%386, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %389 = "linalg.batch_matmul"(%387, %388, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %390 = "linalg.generic"(%389, %168, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %391 = "linalg.generic"(%390, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %392 = "linalg.generic"(%165, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %393 = "linalg.generic"(%391, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %394 = "linalg.generic"(%392, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %395 = "linalg.batch_matmul"(%393, %394, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %396 = "linalg.generic"(%395, %166, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %397 = "linalg.generic"(%396, %372, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %398 = "linalg.generic"(%397, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %399 = "linalg.generic"(%398, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %400 = "linalg.generic"(%399, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %401 = "linalg.generic"(%397, %400, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %402 = "linalg.generic"(%401, %401, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %403 = "linalg.generic"(%402, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %404 = "linalg.generic"(%403, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %405 = "linalg.generic"(%404, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %406 = "linalg.generic"(%405, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %407 = "linalg.generic"(%406, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %408 = "linalg.generic"(%401, %407, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %409 = "linalg.generic"(%408, %163, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %410 = "linalg.generic"(%409, %164, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %411 = "linalg.generic"(%161, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %412 = "linalg.generic"(%410, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %413 = "linalg.generic"(%411, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %414 = "linalg.batch_matmul"(%412, %413, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %415 = "linalg.generic"(%414, %162, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %416 = "linalg.generic"(%159, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %417 = "linalg.generic"(%416, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %418 = "linalg.batch_matmul"(%412, %417, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %419 = "linalg.generic"(%418, %160, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %420 = "tensor.expand_shape"(%419) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %421 = "linalg.generic"(%420, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %422 = "linalg.generic"(%157, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %423 = "linalg.generic"(%422, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %424 = "linalg.batch_matmul"(%412, %423, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %425 = "linalg.generic"(%424, %158, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %426 = "tensor.expand_shape"(%425) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %427 = "linalg.generic"(%426, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %428 = "tensor.expand_shape"(%415) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %429 = "linalg.generic"(%428, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %430 = "linalg.generic"(%421, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %431 = "linalg.generic"(%429, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %432 = "linalg.generic"(%430, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %433 = "tensor.collapse_shape"(%431) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %434 = "tensor.collapse_shape"(%432) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %435 = "linalg.batch_matmul"(%433, %434, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %436 = "tensor.expand_shape"(%435) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %437 = "linalg.generic"(%436, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %438:2 = "linalg.generic"(%437, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %439 = "linalg.generic"(%437, %438#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %440 = "linalg.generic"(%439, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %441 = "linalg.generic"(%440, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %442 = "linalg.generic"(%440, %441, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %443 = "linalg.generic"(%442, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %444 = "linalg.generic"(%427, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %445 = "tensor.collapse_shape"(%443) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %446 = "tensor.collapse_shape"(%444) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %447 = "linalg.batch_matmul"(%445, %446, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %448 = "tensor.expand_shape"(%447) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %449 = "linalg.generic"(%448, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %450 = "tensor.collapse_shape"(%449) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %451 = "linalg.generic"(%155, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %452 = "linalg.generic"(%450, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %453 = "linalg.generic"(%451, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %454 = "linalg.batch_matmul"(%452, %453, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %455 = "linalg.generic"(%454, %156, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %456 = "linalg.generic"(%455, %397, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %457 = "linalg.generic"(%456, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %458 = "linalg.generic"(%457, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %459 = "linalg.generic"(%458, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %460 = "linalg.generic"(%456, %459, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %461 = "linalg.generic"(%460, %460, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %462 = "linalg.generic"(%461, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %463 = "linalg.generic"(%462, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %464 = "linalg.generic"(%463, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %465 = "linalg.generic"(%464, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %466 = "linalg.generic"(%465, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %467 = "linalg.generic"(%460, %466, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %468 = "linalg.generic"(%467, %153, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %469 = "linalg.generic"(%468, %154, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %470 = "linalg.generic"(%151, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %471 = "linalg.generic"(%469, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %472 = "linalg.generic"(%470, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %473 = "linalg.batch_matmul"(%471, %472, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %474 = "linalg.generic"(%473, %152, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %475 = "linalg.generic"(%474, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %476 = "linalg.generic"(%149, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %477 = "linalg.generic"(%475, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %478 = "linalg.generic"(%476, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %479 = "linalg.batch_matmul"(%477, %478, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %480 = "linalg.generic"(%479, %150, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %481 = "linalg.generic"(%480, %456, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %482 = "linalg.generic"(%481, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %483 = "linalg.generic"(%482, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %484 = "linalg.generic"(%483, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %485 = "linalg.generic"(%481, %484, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %486 = "linalg.generic"(%485, %485, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %487 = "linalg.generic"(%486, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %488 = "linalg.generic"(%487, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %489 = "linalg.generic"(%488, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %490 = "linalg.generic"(%489, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %491 = "linalg.generic"(%490, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %492 = "linalg.generic"(%485, %491, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %493 = "linalg.generic"(%492, %147, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %494 = "linalg.generic"(%493, %148, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %495 = "linalg.generic"(%145, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %496 = "linalg.generic"(%494, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %497 = "linalg.generic"(%495, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %498 = "linalg.batch_matmul"(%496, %497, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %499 = "linalg.generic"(%498, %146, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %500 = "linalg.generic"(%143, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %501 = "linalg.generic"(%500, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %502 = "linalg.batch_matmul"(%496, %501, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %503 = "linalg.generic"(%502, %144, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %504 = "tensor.expand_shape"(%503) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %505 = "linalg.generic"(%504, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %506 = "linalg.generic"(%141, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %507 = "linalg.generic"(%506, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %508 = "linalg.batch_matmul"(%496, %507, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %509 = "linalg.generic"(%508, %142, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %510 = "tensor.expand_shape"(%509) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %511 = "linalg.generic"(%510, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %512 = "tensor.expand_shape"(%499) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %513 = "linalg.generic"(%512, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %514 = "linalg.generic"(%505, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %515 = "linalg.generic"(%513, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %516 = "linalg.generic"(%514, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %517 = "tensor.collapse_shape"(%515) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %518 = "tensor.collapse_shape"(%516) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %519 = "linalg.batch_matmul"(%517, %518, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %520 = "tensor.expand_shape"(%519) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %521 = "linalg.generic"(%520, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %522:2 = "linalg.generic"(%521, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %523 = "linalg.generic"(%521, %522#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %524 = "linalg.generic"(%523, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %525 = "linalg.generic"(%524, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %526 = "linalg.generic"(%524, %525, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %527 = "linalg.generic"(%526, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %528 = "linalg.generic"(%511, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %529 = "tensor.collapse_shape"(%527) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %530 = "tensor.collapse_shape"(%528) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %531 = "linalg.batch_matmul"(%529, %530, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %532 = "tensor.expand_shape"(%531) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %533 = "linalg.generic"(%532, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %534 = "tensor.collapse_shape"(%533) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %535 = "linalg.generic"(%139, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %536 = "linalg.generic"(%534, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %537 = "linalg.generic"(%535, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %538 = "linalg.batch_matmul"(%536, %537, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %539 = "linalg.generic"(%538, %140, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %540 = "linalg.generic"(%539, %481, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %541 = "linalg.generic"(%540, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %542 = "linalg.generic"(%541, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %543 = "linalg.generic"(%542, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %544 = "linalg.generic"(%540, %543, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %545 = "linalg.generic"(%544, %544, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %546 = "linalg.generic"(%545, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %547 = "linalg.generic"(%546, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %548 = "linalg.generic"(%547, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %549 = "linalg.generic"(%548, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %550 = "linalg.generic"(%549, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %551 = "linalg.generic"(%544, %550, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %552 = "linalg.generic"(%551, %137, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %553 = "linalg.generic"(%552, %138, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %554 = "linalg.generic"(%135, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %555 = "linalg.generic"(%553, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %556 = "linalg.generic"(%554, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %557 = "linalg.batch_matmul"(%555, %556, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %558 = "linalg.generic"(%557, %136, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %559 = "linalg.generic"(%558, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %560 = "linalg.generic"(%133, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %561 = "linalg.generic"(%559, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %562 = "linalg.generic"(%560, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %563 = "linalg.batch_matmul"(%561, %562, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %564 = "linalg.generic"(%563, %134, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %565 = "linalg.generic"(%564, %540, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %566 = "linalg.generic"(%565, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %567 = "linalg.generic"(%566, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %568 = "linalg.generic"(%567, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %569 = "linalg.generic"(%565, %568, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %570 = "linalg.generic"(%569, %569, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %571 = "linalg.generic"(%570, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %572 = "linalg.generic"(%571, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %573 = "linalg.generic"(%572, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %574 = "linalg.generic"(%573, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %575 = "linalg.generic"(%574, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %576 = "linalg.generic"(%569, %575, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %577 = "linalg.generic"(%576, %131, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %578 = "linalg.generic"(%577, %132, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %579 = "linalg.generic"(%129, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %580 = "linalg.generic"(%578, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %581 = "linalg.generic"(%579, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %582 = "linalg.batch_matmul"(%580, %581, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %583 = "linalg.generic"(%582, %130, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %584 = "linalg.generic"(%127, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %585 = "linalg.generic"(%584, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %586 = "linalg.batch_matmul"(%580, %585, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %587 = "linalg.generic"(%586, %128, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %588 = "tensor.expand_shape"(%587) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %589 = "linalg.generic"(%588, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %590 = "linalg.generic"(%125, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %591 = "linalg.generic"(%590, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %592 = "linalg.batch_matmul"(%580, %591, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %593 = "linalg.generic"(%592, %126, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %594 = "tensor.expand_shape"(%593) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %595 = "linalg.generic"(%594, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %596 = "tensor.expand_shape"(%583) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %597 = "linalg.generic"(%596, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %598 = "linalg.generic"(%589, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %599 = "linalg.generic"(%597, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %600 = "linalg.generic"(%598, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %601 = "tensor.collapse_shape"(%599) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %602 = "tensor.collapse_shape"(%600) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %603 = "linalg.batch_matmul"(%601, %602, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %604 = "tensor.expand_shape"(%603) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %605 = "linalg.generic"(%604, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %606:2 = "linalg.generic"(%605, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %607 = "linalg.generic"(%605, %606#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %608 = "linalg.generic"(%607, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %609 = "linalg.generic"(%608, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %610 = "linalg.generic"(%608, %609, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %611 = "linalg.generic"(%610, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %612 = "linalg.generic"(%595, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %613 = "tensor.collapse_shape"(%611) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %614 = "tensor.collapse_shape"(%612) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %615 = "linalg.batch_matmul"(%613, %614, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %616 = "tensor.expand_shape"(%615) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %617 = "linalg.generic"(%616, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %618 = "tensor.collapse_shape"(%617) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %619 = "linalg.generic"(%123, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %620 = "linalg.generic"(%618, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %621 = "linalg.generic"(%619, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %622 = "linalg.batch_matmul"(%620, %621, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %623 = "linalg.generic"(%622, %124, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %624 = "linalg.generic"(%623, %565, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %625 = "linalg.generic"(%624, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %626 = "linalg.generic"(%625, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %627 = "linalg.generic"(%626, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %628 = "linalg.generic"(%624, %627, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %629 = "linalg.generic"(%628, %628, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %630 = "linalg.generic"(%629, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %631 = "linalg.generic"(%630, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %632 = "linalg.generic"(%631, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %633 = "linalg.generic"(%632, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %634 = "linalg.generic"(%633, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %635 = "linalg.generic"(%628, %634, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %636 = "linalg.generic"(%635, %121, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %637 = "linalg.generic"(%636, %122, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %638 = "linalg.generic"(%119, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %639 = "linalg.generic"(%637, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %640 = "linalg.generic"(%638, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %641 = "linalg.batch_matmul"(%639, %640, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %642 = "linalg.generic"(%641, %120, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %643 = "linalg.generic"(%642, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %644 = "linalg.generic"(%117, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %645 = "linalg.generic"(%643, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %646 = "linalg.generic"(%644, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %647 = "linalg.batch_matmul"(%645, %646, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %648 = "linalg.generic"(%647, %118, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %649 = "linalg.generic"(%648, %624, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %650 = "linalg.generic"(%649, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %651 = "linalg.generic"(%650, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %652 = "linalg.generic"(%651, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %653 = "linalg.generic"(%649, %652, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %654 = "linalg.generic"(%653, %653, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %655 = "linalg.generic"(%654, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %656 = "linalg.generic"(%655, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %657 = "linalg.generic"(%656, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %658 = "linalg.generic"(%657, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %659 = "linalg.generic"(%658, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %660 = "linalg.generic"(%653, %659, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %661 = "linalg.generic"(%660, %115, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %662 = "linalg.generic"(%661, %116, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %663 = "linalg.generic"(%113, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %664 = "linalg.generic"(%662, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %665 = "linalg.generic"(%663, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %666 = "linalg.batch_matmul"(%664, %665, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %667 = "linalg.generic"(%666, %114, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %668 = "linalg.generic"(%111, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %669 = "linalg.generic"(%668, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %670 = "linalg.batch_matmul"(%664, %669, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %671 = "linalg.generic"(%670, %112, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %672 = "tensor.expand_shape"(%671) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %673 = "linalg.generic"(%672, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %674 = "linalg.generic"(%109, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %675 = "linalg.generic"(%674, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %676 = "linalg.batch_matmul"(%664, %675, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %677 = "linalg.generic"(%676, %110, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %678 = "tensor.expand_shape"(%677) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %679 = "linalg.generic"(%678, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %680 = "tensor.expand_shape"(%667) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %681 = "linalg.generic"(%680, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %682 = "linalg.generic"(%673, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %683 = "linalg.generic"(%681, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %684 = "linalg.generic"(%682, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %685 = "tensor.collapse_shape"(%683) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %686 = "tensor.collapse_shape"(%684) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %687 = "linalg.batch_matmul"(%685, %686, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %688 = "tensor.expand_shape"(%687) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %689 = "linalg.generic"(%688, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %690:2 = "linalg.generic"(%689, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %691 = "linalg.generic"(%689, %690#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %692 = "linalg.generic"(%691, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %693 = "linalg.generic"(%692, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %694 = "linalg.generic"(%692, %693, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %695 = "linalg.generic"(%694, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %696 = "linalg.generic"(%679, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %697 = "tensor.collapse_shape"(%695) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %698 = "tensor.collapse_shape"(%696) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %699 = "linalg.batch_matmul"(%697, %698, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %700 = "tensor.expand_shape"(%699) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %701 = "linalg.generic"(%700, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %702 = "tensor.collapse_shape"(%701) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %703 = "linalg.generic"(%107, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %704 = "linalg.generic"(%702, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %705 = "linalg.generic"(%703, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %706 = "linalg.batch_matmul"(%704, %705, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %707 = "linalg.generic"(%706, %108, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %708 = "linalg.generic"(%707, %649, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %709 = "linalg.generic"(%708, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %710 = "linalg.generic"(%709, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %711 = "linalg.generic"(%710, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %712 = "linalg.generic"(%708, %711, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %713 = "linalg.generic"(%712, %712, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %714 = "linalg.generic"(%713, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %715 = "linalg.generic"(%714, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %716 = "linalg.generic"(%715, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %717 = "linalg.generic"(%716, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %718 = "linalg.generic"(%717, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %719 = "linalg.generic"(%712, %718, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %720 = "linalg.generic"(%719, %105, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %721 = "linalg.generic"(%720, %106, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %722 = "linalg.generic"(%103, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %723 = "linalg.generic"(%721, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %724 = "linalg.generic"(%722, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %725 = "linalg.batch_matmul"(%723, %724, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %726 = "linalg.generic"(%725, %104, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %727 = "linalg.generic"(%726, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %728 = "linalg.generic"(%101, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %729 = "linalg.generic"(%727, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %730 = "linalg.generic"(%728, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %731 = "linalg.batch_matmul"(%729, %730, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %732 = "linalg.generic"(%731, %102, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %733 = "linalg.generic"(%732, %708, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %734 = "linalg.generic"(%733, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %735 = "linalg.generic"(%734, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %736 = "linalg.generic"(%735, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %737 = "linalg.generic"(%733, %736, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %738 = "linalg.generic"(%737, %737, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %739 = "linalg.generic"(%738, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %740 = "linalg.generic"(%739, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %741 = "linalg.generic"(%740, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %742 = "linalg.generic"(%741, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %743 = "linalg.generic"(%742, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %744 = "linalg.generic"(%737, %743, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %745 = "linalg.generic"(%744, %99, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %746 = "linalg.generic"(%745, %100, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %747 = "linalg.generic"(%97, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %748 = "linalg.generic"(%746, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %749 = "linalg.generic"(%747, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %750 = "linalg.batch_matmul"(%748, %749, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %751 = "linalg.generic"(%750, %98, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %752 = "linalg.generic"(%95, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %753 = "linalg.generic"(%752, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %754 = "linalg.batch_matmul"(%748, %753, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %755 = "linalg.generic"(%754, %96, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %756 = "tensor.expand_shape"(%755) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %757 = "linalg.generic"(%756, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %758 = "linalg.generic"(%93, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %759 = "linalg.generic"(%758, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %760 = "linalg.batch_matmul"(%748, %759, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %761 = "linalg.generic"(%760, %94, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %762 = "tensor.expand_shape"(%761) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %763 = "linalg.generic"(%762, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %764 = "tensor.expand_shape"(%751) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %765 = "linalg.generic"(%764, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %766 = "linalg.generic"(%757, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %767 = "linalg.generic"(%765, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %768 = "linalg.generic"(%766, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %769 = "tensor.collapse_shape"(%767) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %770 = "tensor.collapse_shape"(%768) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %771 = "linalg.batch_matmul"(%769, %770, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %772 = "tensor.expand_shape"(%771) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %773 = "linalg.generic"(%772, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %774:2 = "linalg.generic"(%773, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %775 = "linalg.generic"(%773, %774#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %776 = "linalg.generic"(%775, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %777 = "linalg.generic"(%776, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %778 = "linalg.generic"(%776, %777, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %779 = "linalg.generic"(%778, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %780 = "linalg.generic"(%763, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %781 = "tensor.collapse_shape"(%779) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %782 = "tensor.collapse_shape"(%780) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %783 = "linalg.batch_matmul"(%781, %782, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %784 = "tensor.expand_shape"(%783) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %785 = "linalg.generic"(%784, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %786 = "tensor.collapse_shape"(%785) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %787 = "linalg.generic"(%91, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %788 = "linalg.generic"(%786, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %789 = "linalg.generic"(%787, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %790 = "linalg.batch_matmul"(%788, %789, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %791 = "linalg.generic"(%790, %92, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %792 = "linalg.generic"(%791, %733, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %793 = "linalg.generic"(%792, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %794 = "linalg.generic"(%793, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %795 = "linalg.generic"(%794, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %796 = "linalg.generic"(%792, %795, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %797 = "linalg.generic"(%796, %796, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %798 = "linalg.generic"(%797, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %799 = "linalg.generic"(%798, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %800 = "linalg.generic"(%799, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %801 = "linalg.generic"(%800, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %802 = "linalg.generic"(%801, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %803 = "linalg.generic"(%796, %802, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %804 = "linalg.generic"(%803, %89, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %805 = "linalg.generic"(%804, %90, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %806 = "linalg.generic"(%87, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %807 = "linalg.generic"(%805, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %808 = "linalg.generic"(%806, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %809 = "linalg.batch_matmul"(%807, %808, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %810 = "linalg.generic"(%809, %88, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %811 = "linalg.generic"(%810, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %812 = "linalg.generic"(%85, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %813 = "linalg.generic"(%811, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %814 = "linalg.generic"(%812, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %815 = "linalg.batch_matmul"(%813, %814, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %816 = "linalg.generic"(%815, %86, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %817 = "linalg.generic"(%816, %792, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %818 = "linalg.generic"(%817, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %819 = "linalg.generic"(%818, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %820 = "linalg.generic"(%819, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %821 = "linalg.generic"(%817, %820, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %822 = "linalg.generic"(%821, %821, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %823 = "linalg.generic"(%822, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %824 = "linalg.generic"(%823, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %825 = "linalg.generic"(%824, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %826 = "linalg.generic"(%825, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %827 = "linalg.generic"(%826, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %828 = "linalg.generic"(%821, %827, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %829 = "linalg.generic"(%828, %83, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %830 = "linalg.generic"(%829, %84, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %831 = "linalg.generic"(%81, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %832 = "linalg.generic"(%830, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %833 = "linalg.generic"(%831, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %834 = "linalg.batch_matmul"(%832, %833, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %835 = "linalg.generic"(%834, %82, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %836 = "linalg.generic"(%79, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %837 = "linalg.generic"(%836, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %838 = "linalg.batch_matmul"(%832, %837, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %839 = "linalg.generic"(%838, %80, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %840 = "tensor.expand_shape"(%839) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %841 = "linalg.generic"(%840, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %842 = "linalg.generic"(%77, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %843 = "linalg.generic"(%842, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %844 = "linalg.batch_matmul"(%832, %843, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %845 = "linalg.generic"(%844, %78, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %846 = "tensor.expand_shape"(%845) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %847 = "linalg.generic"(%846, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %848 = "tensor.expand_shape"(%835) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %849 = "linalg.generic"(%848, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %850 = "linalg.generic"(%841, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %851 = "linalg.generic"(%849, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %852 = "linalg.generic"(%850, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %853 = "tensor.collapse_shape"(%851) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %854 = "tensor.collapse_shape"(%852) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %855 = "linalg.batch_matmul"(%853, %854, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %856 = "tensor.expand_shape"(%855) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %857 = "linalg.generic"(%856, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %858:2 = "linalg.generic"(%857, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %859 = "linalg.generic"(%857, %858#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %860 = "linalg.generic"(%859, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %861 = "linalg.generic"(%860, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %862 = "linalg.generic"(%860, %861, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %863 = "linalg.generic"(%862, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %864 = "linalg.generic"(%847, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %865 = "tensor.collapse_shape"(%863) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %866 = "tensor.collapse_shape"(%864) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %867 = "linalg.batch_matmul"(%865, %866, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %868 = "tensor.expand_shape"(%867) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %869 = "linalg.generic"(%868, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %870 = "tensor.collapse_shape"(%869) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %871 = "linalg.generic"(%75, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %872 = "linalg.generic"(%870, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %873 = "linalg.generic"(%871, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %874 = "linalg.batch_matmul"(%872, %873, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %875 = "linalg.generic"(%874, %76, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %876 = "linalg.generic"(%875, %817, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %877 = "linalg.generic"(%876, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %878 = "linalg.generic"(%877, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %879 = "linalg.generic"(%878, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %880 = "linalg.generic"(%876, %879, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %881 = "linalg.generic"(%880, %880, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %882 = "linalg.generic"(%881, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %883 = "linalg.generic"(%882, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %884 = "linalg.generic"(%883, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %885 = "linalg.generic"(%884, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %886 = "linalg.generic"(%885, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %887 = "linalg.generic"(%880, %886, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %888 = "linalg.generic"(%887, %73, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %889 = "linalg.generic"(%888, %74, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %890 = "linalg.generic"(%71, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %891 = "linalg.generic"(%889, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %892 = "linalg.generic"(%890, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %893 = "linalg.batch_matmul"(%891, %892, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %894 = "linalg.generic"(%893, %72, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %895 = "linalg.generic"(%894, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %896 = "linalg.generic"(%69, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %897 = "linalg.generic"(%895, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %898 = "linalg.generic"(%896, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %899 = "linalg.batch_matmul"(%897, %898, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %900 = "linalg.generic"(%899, %70, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %901 = "linalg.generic"(%900, %876, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %902 = "linalg.generic"(%901, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %903 = "linalg.generic"(%902, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %904 = "linalg.generic"(%903, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %905 = "linalg.generic"(%901, %904, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %906 = "linalg.generic"(%905, %905, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %907 = "linalg.generic"(%906, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %908 = "linalg.generic"(%907, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %909 = "linalg.generic"(%908, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %910 = "linalg.generic"(%909, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %911 = "linalg.generic"(%910, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %912 = "linalg.generic"(%905, %911, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %913 = "linalg.generic"(%912, %67, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %914 = "linalg.generic"(%913, %68, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %915 = "linalg.generic"(%65, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %916 = "linalg.generic"(%914, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %917 = "linalg.generic"(%915, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %918 = "linalg.batch_matmul"(%916, %917, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %919 = "linalg.generic"(%918, %66, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %920 = "linalg.generic"(%63, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %921 = "linalg.generic"(%920, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %922 = "linalg.batch_matmul"(%916, %921, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %923 = "linalg.generic"(%922, %64, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %924 = "tensor.expand_shape"(%923) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %925 = "linalg.generic"(%924, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %926 = "linalg.generic"(%61, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %927 = "linalg.generic"(%926, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %928 = "linalg.batch_matmul"(%916, %927, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %929 = "linalg.generic"(%928, %62, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %930 = "tensor.expand_shape"(%929) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %931 = "linalg.generic"(%930, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %932 = "tensor.expand_shape"(%919) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %933 = "linalg.generic"(%932, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %934 = "linalg.generic"(%925, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %935 = "linalg.generic"(%933, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %936 = "linalg.generic"(%934, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %937 = "tensor.collapse_shape"(%935) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %938 = "tensor.collapse_shape"(%936) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %939 = "linalg.batch_matmul"(%937, %938, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %940 = "tensor.expand_shape"(%939) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %941 = "linalg.generic"(%940, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %942:2 = "linalg.generic"(%941, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %943 = "linalg.generic"(%941, %942#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %944 = "linalg.generic"(%943, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %945 = "linalg.generic"(%944, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %946 = "linalg.generic"(%944, %945, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %947 = "linalg.generic"(%946, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %948 = "linalg.generic"(%931, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %949 = "tensor.collapse_shape"(%947) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %950 = "tensor.collapse_shape"(%948) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %951 = "linalg.batch_matmul"(%949, %950, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %952 = "tensor.expand_shape"(%951) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %953 = "linalg.generic"(%952, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %954 = "tensor.collapse_shape"(%953) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %955 = "linalg.generic"(%59, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %956 = "linalg.generic"(%954, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %957 = "linalg.generic"(%955, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %958 = "linalg.batch_matmul"(%956, %957, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %959 = "linalg.generic"(%958, %60, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %960 = "linalg.generic"(%959, %901, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %961 = "linalg.generic"(%960, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %962 = "linalg.generic"(%961, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %963 = "linalg.generic"(%962, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %964 = "linalg.generic"(%960, %963, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %965 = "linalg.generic"(%964, %964, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %966 = "linalg.generic"(%965, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %967 = "linalg.generic"(%966, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %968 = "linalg.generic"(%967, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %969 = "linalg.generic"(%968, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %970 = "linalg.generic"(%969, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %971 = "linalg.generic"(%964, %970, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %972 = "linalg.generic"(%971, %57, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %973 = "linalg.generic"(%972, %58, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %974 = "linalg.generic"(%55, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %975 = "linalg.generic"(%973, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %976 = "linalg.generic"(%974, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %977 = "linalg.batch_matmul"(%975, %976, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %978 = "linalg.generic"(%977, %56, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %979 = "linalg.generic"(%978, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %980 = "linalg.generic"(%53, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %981 = "linalg.generic"(%979, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %982 = "linalg.generic"(%980, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %983 = "linalg.batch_matmul"(%981, %982, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %984 = "linalg.generic"(%983, %54, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %985 = "linalg.generic"(%984, %960, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %986 = "linalg.generic"(%985, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %987 = "linalg.generic"(%986, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %988 = "linalg.generic"(%987, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %989 = "linalg.generic"(%985, %988, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %990 = "linalg.generic"(%989, %989, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %991 = "linalg.generic"(%990, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %992 = "linalg.generic"(%991, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %993 = "linalg.generic"(%992, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %994 = "linalg.generic"(%993, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %995 = "linalg.generic"(%994, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %996 = "linalg.generic"(%989, %995, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %997 = "linalg.generic"(%996, %51, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %998 = "linalg.generic"(%997, %52, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %999 = "linalg.generic"(%49, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1000 = "linalg.generic"(%998, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1001 = "linalg.generic"(%999, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1002 = "linalg.batch_matmul"(%1000, %1001, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1003 = "linalg.generic"(%1002, %50, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1004 = "linalg.generic"(%47, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1005 = "linalg.generic"(%1004, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1006 = "linalg.batch_matmul"(%1000, %1005, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1007 = "linalg.generic"(%1006, %48, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1008 = "tensor.expand_shape"(%1007) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1009 = "linalg.generic"(%1008, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1010 = "linalg.generic"(%45, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1011 = "linalg.generic"(%1010, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1012 = "linalg.batch_matmul"(%1000, %1011, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1013 = "linalg.generic"(%1012, %46, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1014 = "tensor.expand_shape"(%1013) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1015 = "linalg.generic"(%1014, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1016 = "tensor.expand_shape"(%1003) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1017 = "linalg.generic"(%1016, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1018 = "linalg.generic"(%1009, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %1019 = "linalg.generic"(%1017, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1020 = "linalg.generic"(%1018, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %1021 = "tensor.collapse_shape"(%1019) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %1022 = "tensor.collapse_shape"(%1020) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %1023 = "linalg.batch_matmul"(%1021, %1022, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %1024 = "tensor.expand_shape"(%1023) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1025 = "linalg.generic"(%1024, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1026:2 = "linalg.generic"(%1025, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %1027 = "linalg.generic"(%1025, %1026#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1028 = "linalg.generic"(%1027, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1029 = "linalg.generic"(%1028, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %1030 = "linalg.generic"(%1028, %1029, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1031 = "linalg.generic"(%1030, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1032 = "linalg.generic"(%1015, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1033 = "tensor.collapse_shape"(%1031) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %1034 = "tensor.collapse_shape"(%1032) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %1035 = "linalg.batch_matmul"(%1033, %1034, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %1036 = "tensor.expand_shape"(%1035) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1037 = "linalg.generic"(%1036, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %1038 = "tensor.collapse_shape"(%1037) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %1039 = "linalg.generic"(%43, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1040 = "linalg.generic"(%1038, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1041 = "linalg.generic"(%1039, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1042 = "linalg.batch_matmul"(%1040, %1041, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1043 = "linalg.generic"(%1042, %44, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1044 = "linalg.generic"(%1043, %985, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1045 = "linalg.generic"(%1044, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1046 = "linalg.generic"(%1045, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1047 = "linalg.generic"(%1046, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1048 = "linalg.generic"(%1044, %1047, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1049 = "linalg.generic"(%1048, %1048, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1050 = "linalg.generic"(%1049, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1051 = "linalg.generic"(%1050, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1052 = "linalg.generic"(%1051, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1053 = "linalg.generic"(%1052, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1054 = "linalg.generic"(%1053, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1055 = "linalg.generic"(%1048, %1054, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1056 = "linalg.generic"(%1055, %41, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1057 = "linalg.generic"(%1056, %42, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1058 = "linalg.generic"(%39, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %1059 = "linalg.generic"(%1057, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1060 = "linalg.generic"(%1058, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %1061 = "linalg.batch_matmul"(%1059, %1060, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1062 = "linalg.generic"(%1061, %40, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1063 = "linalg.generic"(%1062, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1064 = "linalg.generic"(%37, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %1065 = "linalg.generic"(%1063, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1066 = "linalg.generic"(%1064, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %1067 = "linalg.batch_matmul"(%1065, %1066, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1068 = "linalg.generic"(%1067, %38, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1069 = "linalg.generic"(%1068, %1044, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1070 = "linalg.generic"(%1069, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1071 = "linalg.generic"(%1070, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1072 = "linalg.generic"(%1071, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1073 = "linalg.generic"(%1069, %1072, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1074 = "linalg.generic"(%1073, %1073, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1075 = "linalg.generic"(%1074, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1076 = "linalg.generic"(%1075, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1077 = "linalg.generic"(%1076, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1078 = "linalg.generic"(%1077, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1079 = "linalg.generic"(%1078, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1080 = "linalg.generic"(%1073, %1079, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1081 = "linalg.generic"(%1080, %35, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1082 = "linalg.generic"(%1081, %36, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1083 = "linalg.generic"(%33, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1084 = "linalg.generic"(%1082, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1085 = "linalg.generic"(%1083, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1086 = "linalg.batch_matmul"(%1084, %1085, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1087 = "linalg.generic"(%1086, %34, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1088 = "linalg.generic"(%31, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1089 = "linalg.generic"(%1088, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1090 = "linalg.batch_matmul"(%1084, %1089, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1091 = "linalg.generic"(%1090, %32, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1092 = "tensor.expand_shape"(%1091) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1093 = "linalg.generic"(%1092, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1094 = "linalg.generic"(%29, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1095 = "linalg.generic"(%1094, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1096 = "linalg.batch_matmul"(%1084, %1095, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1097 = "linalg.generic"(%1096, %30, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1098 = "tensor.expand_shape"(%1097) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1099 = "linalg.generic"(%1098, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1100 = "tensor.expand_shape"(%1087) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1101 = "linalg.generic"(%1100, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1102 = "linalg.generic"(%1093, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %1103 = "linalg.generic"(%1101, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1104 = "linalg.generic"(%1102, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %1105 = "tensor.collapse_shape"(%1103) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %1106 = "tensor.collapse_shape"(%1104) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %1107 = "linalg.batch_matmul"(%1105, %1106, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %1108 = "tensor.expand_shape"(%1107) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1109 = "linalg.generic"(%1108, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1110:2 = "linalg.generic"(%1109, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %1111 = "linalg.generic"(%1109, %1110#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1112 = "linalg.generic"(%1111, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1113 = "linalg.generic"(%1112, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %1114 = "linalg.generic"(%1112, %1113, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1115 = "linalg.generic"(%1114, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1116 = "linalg.generic"(%1099, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1117 = "tensor.collapse_shape"(%1115) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %1118 = "tensor.collapse_shape"(%1116) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %1119 = "linalg.batch_matmul"(%1117, %1118, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %1120 = "tensor.expand_shape"(%1119) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1121 = "linalg.generic"(%1120, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %1122 = "tensor.collapse_shape"(%1121) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %1123 = "linalg.generic"(%27, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1124 = "linalg.generic"(%1122, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1125 = "linalg.generic"(%1123, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1126 = "linalg.batch_matmul"(%1124, %1125, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1127 = "linalg.generic"(%1126, %28, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1128 = "linalg.generic"(%1127, %1069, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1129 = "linalg.generic"(%1128, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1130 = "linalg.generic"(%1129, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1131 = "linalg.generic"(%1130, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1132 = "linalg.generic"(%1128, %1131, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1133 = "linalg.generic"(%1132, %1132, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1134 = "linalg.generic"(%1133, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1135 = "linalg.generic"(%1134, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1136 = "linalg.generic"(%1135, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1137 = "linalg.generic"(%1136, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1138 = "linalg.generic"(%1137, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1139 = "linalg.generic"(%1132, %1138, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1140 = "linalg.generic"(%1139, %25, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1141 = "linalg.generic"(%1140, %26, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1142 = "linalg.generic"(%23, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %1143 = "linalg.generic"(%1141, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1144 = "linalg.generic"(%1142, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %1145 = "linalg.batch_matmul"(%1143, %1144, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1146 = "linalg.generic"(%1145, %24, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1147 = "linalg.generic"(%1146, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1148 = "linalg.generic"(%21, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %1149 = "linalg.generic"(%1147, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1150 = "linalg.generic"(%1148, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %1151 = "linalg.batch_matmul"(%1149, %1150, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1152 = "linalg.generic"(%1151, %22, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1153 = "linalg.generic"(%1152, %1128, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1154 = "linalg.generic"(%1153, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1155 = "linalg.generic"(%1154, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1156 = "linalg.generic"(%1155, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1157 = "linalg.generic"(%1153, %1156, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1158 = "linalg.generic"(%1157, %1157, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1159 = "linalg.generic"(%1158, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1160 = "linalg.generic"(%1159, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1161 = "linalg.generic"(%1160, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1162 = "linalg.generic"(%1161, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1163 = "linalg.generic"(%1162, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1164 = "linalg.generic"(%1157, %1163, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1165 = "linalg.generic"(%1164, %19, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1166 = "linalg.generic"(%1165, %20, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1167 = "linalg.generic"(%17, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1168 = "linalg.generic"(%1166, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1169 = "linalg.generic"(%1167, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1170 = "linalg.batch_matmul"(%1168, %1169, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1171 = "linalg.generic"(%1170, %18, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1172 = "linalg.generic"(%15, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1173 = "linalg.generic"(%1172, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1174 = "linalg.batch_matmul"(%1168, %1173, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1175 = "linalg.generic"(%1174, %16, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1176 = "tensor.expand_shape"(%1175) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1177 = "linalg.generic"(%1176, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1178 = "linalg.generic"(%13, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1179 = "linalg.generic"(%1178, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1180 = "linalg.batch_matmul"(%1168, %1179, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1181 = "linalg.generic"(%1180, %14, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1182 = "tensor.expand_shape"(%1181) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1183 = "linalg.generic"(%1182, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1184 = "tensor.expand_shape"(%1171) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x768xf32>) -> tensor<1x197x12x64xf32>
    %1185 = "linalg.generic"(%1184, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x12x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1186 = "linalg.generic"(%1177, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map13], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %1187 = "linalg.generic"(%1185, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1188 = "linalg.generic"(%1186, %244) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x64x197xf32>, tensor<1x12x64x197xf32>) -> tensor<1x12x64x197xf32>
    %1189 = "tensor.collapse_shape"(%1187) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %1190 = "tensor.collapse_shape"(%1188) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    %1191 = "linalg.batch_matmul"(%1189, %1190, %251) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x64xf32>, tensor<12x64x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %1192 = "tensor.expand_shape"(%1191) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1193 = "linalg.generic"(%1192, %189, %254) ({
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f32):
      %1260 = "arith.truncf"(%arg2) : (f64) -> f32
      %1261 = "arith.divf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map14, #map15, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<f64>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1194:2 = "linalg.generic"(%1193, %259, %257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %1260 = "linalg.index"() {dim = 3 : i64} : () -> index
      %1261 = "arith.index_cast"(%1260) : (index) -> i64
      %1262 = "arith.maxf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.cmpf"(%arg1, %arg2) {predicate = 2 : i64} : (f32, f32) -> i1
      %1264 = "arith.select"(%1263, %1261, %arg3) : (i1, i64, i64) -> i64
      "linalg.yield"(%1262, %1264) : (f32, i64) -> ()
    }) {indexing_maps = [#map11, #map16, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 2>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>) -> (tensor<1x12x197x1xf32>, tensor<1x12x197x1xi64>)
    %1195 = "linalg.generic"(%1193, %1194#0, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1196 = "linalg.generic"(%1195, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.exp"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1197 = "linalg.generic"(%1196, %263) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map11, #map16], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>) -> tensor<1x12x197x1xf32>
    %1198 = "linalg.generic"(%1196, %1197, %254) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.divf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map14, #map17, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x1xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1199 = "linalg.generic"(%1198, %254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %1200 = "linalg.generic"(%1183, %234) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map14, #map11], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1201 = "tensor.collapse_shape"(%1199) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    %1202 = "tensor.collapse_shape"(%1200) {reassociation = [[0, 1], [2], [3]]} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    %1203 = "linalg.batch_matmul"(%1201, %1202, %271) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<12x197x197xf32>, tensor<12x197x64xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %1204 = "tensor.expand_shape"(%1203) {reassociation = [[0, 1], [2], [3]]} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1205 = "linalg.generic"(%1204, %274) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x12x197x64xf32>, tensor<1x197x12x64xf32>) -> tensor<1x197x12x64xf32>
    %1206 = "tensor.collapse_shape"(%1205) {reassociation = [[0], [1], [2, 3]]} : (tensor<1x197x12x64xf32>) -> tensor<1x197x768xf32>
    %1207 = "linalg.generic"(%11, %221) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<768x768xf32>
    %1208 = "linalg.generic"(%1206, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1209 = "linalg.generic"(%1207, %224) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x768xf32>, tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %1210 = "linalg.batch_matmul"(%1208, %1209, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1211 = "linalg.generic"(%1210, %12, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1212 = "linalg.generic"(%1211, %1153, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1213 = "linalg.generic"(%1212, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1214 = "linalg.generic"(%1213, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1215 = "linalg.generic"(%1214, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1216 = "linalg.generic"(%1212, %1215, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1217 = "linalg.generic"(%1216, %1216, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1218 = "linalg.generic"(%1217, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1219 = "linalg.generic"(%1218, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1220 = "linalg.generic"(%1219, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1221 = "linalg.generic"(%1220, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1222 = "linalg.generic"(%1221, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1223 = "linalg.generic"(%1216, %1222, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1224 = "linalg.generic"(%1223, %9, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1225 = "linalg.generic"(%1224, %10, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1226 = "linalg.generic"(%7, %296) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<768x3072xf32>) -> tensor<768x3072xf32>
    %1227 = "linalg.generic"(%1225, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1228 = "linalg.generic"(%1226, %299) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %1229 = "linalg.batch_matmul"(%1227, %1228, %302) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1230 = "linalg.generic"(%1229, %8, %301) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1231 = "linalg.generic"(%1230, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %201) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "math.erf"(%1260) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      %1262 = "arith.addf"(%1261, %200) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1263 = "arith.mulf"(%1262, %202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1264 = "arith.mulf"(%arg1, %1263) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1264) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1232 = "linalg.generic"(%5, %306) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<3072x768xf32>
    %1233 = "linalg.generic"(%1231, %301) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %1234 = "linalg.generic"(%1232, %309) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<3072x768xf32>, tensor<1x3072x768xf32>) -> tensor<1x3072x768xf32>
    %1235 = "linalg.batch_matmul"(%1233, %1234, %226) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x3072xf32>, tensor<1x3072x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1236 = "linalg.generic"(%1235, %6, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1237 = "linalg.generic"(%1236, %1212, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1238 = "linalg.generic"(%1237, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1239 = "linalg.generic"(%1238, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1240 = "linalg.generic"(%1239, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1241 = "linalg.generic"(%1237, %1240, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.subf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1242 = "linalg.generic"(%1241, %1241, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1243 = "linalg.generic"(%1242, %206) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1244 = "linalg.generic"(%1243, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.divf"(%arg1, %204) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1245 = "linalg.generic"(%1244, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "arith.truncf"(%203) : (f64) -> f32
      %1261 = "arith.addf"(%arg1, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1246 = "linalg.generic"(%1245, %205) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1260 = "math.rsqrt"(%arg1) {fastmath = #arith.fastmath<none>} : (f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %1247 = "linalg.generic"(%1246, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map2, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x1xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1248 = "linalg.generic"(%1241, %1247, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1249 = "linalg.generic"(%1248, %3, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1250 = "linalg.generic"(%1249, %4, %209) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1251 = "tensor.empty"() : () -> tensor<768x1000xf32>
    %1252 = "linalg.generic"(%1, %1251) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1000x768xf32>, tensor<768x1000xf32>) -> tensor<768x1000xf32>
    %1253 = "linalg.generic"(%1250, %209) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map3, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %1254 = "tensor.empty"() : () -> tensor<1x768x1000xf32>
    %1255 = "linalg.generic"(%1252, %1254) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {indexing_maps = [#map7, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 1, 1>} : (tensor<768x1000xf32>, tensor<1x768x1000xf32>) -> tensor<1x768x1000xf32>
    %1256 = "tensor.empty"() : () -> tensor<1x197x1000xf32>
    %1257 = "linalg.fill"(%198, %1256) ({
    ^bb0(%arg1: f32, %arg2: f32):
      "linalg.yield"(%arg1) : (f32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 1>} : (f32, tensor<1x197x1000xf32>) -> tensor<1x197x1000xf32>
    %1258 = "linalg.batch_matmul"(%1253, %1255, %1257) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.mulf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %1261 = "arith.addf"(%arg3, %1260) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1261) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map8, #map9, #map10], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x768xf32>, tensor<1x768x1000xf32>, tensor<1x197x1000xf32>) -> tensor<1x197x1000xf32>
    %1259 = "linalg.generic"(%1258, %2, %1256) ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %1260 = "arith.addf"(%arg1, %arg2) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%1260) : (f32) -> ()
    }) {indexing_maps = [#map3, #map4, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x197x1000xf32>, tensor<1000xf32>, tensor<1x197x1000xf32>) -> tensor<1x197x1000xf32>
    "func.return"(%1259) : (tensor<1x197x1000xf32>) -> ()
  }) {function_type = (tensor<1x197x768xf32>) -> tensor<1x197x1000xf32>, sym_name = "forward"} : () -> ()
}) {torch.debug_module_name = "vit"} : () -> ()
