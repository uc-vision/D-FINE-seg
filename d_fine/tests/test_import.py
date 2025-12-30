"""Test import of all files to check syntax, path"""

from ucvision_utility.testing import check_imports


def test_imports():
  check_imports(
    "d_fine",
    ignore_modules=[
      "d_fine.infer.trt_model",
      "d_fine.infer.trt_alr",
      "d_fine.infer.onnx_model",
      "d_fine.infer.ov_model",
    ],
  )


if __name__ == "__main__":
  test_imports()
