# import nbformat
# from pathlib import Path
# from nbconvert.preprocessors import ExecutePreprocessor
# import unittest
# from typing import Callable
#
#
# DIR_ROOT = Path(__file__).parent.parent
# DIR_EXAMPLES = DIR_ROOT / "examples"
#
#
# class NotebookRunner:
#     DEFAULT_VERSION = 4
#     DEFAULT_KERNEL = "python3"
#     DEFAULT_TIMEOUT = 600
#
#     def __init__(self, path: Path) -> None:
#         self.path = path
#         self.__notebook = None
#
#     def __get_outputs(self, notebook, filter: Callable) -> dict[int, dict[int, list]]:
#         outputs: dict[int, dict[int, list]] = dict()
#         for i, cell in enumerate(notebook.cells):
#             if i not in outputs:
#                 outputs[i] = dict()
#             if "outputs" in cell:
#                 for j, output in enumerate(cell["outputs"]):
#                     if j not in outputs[i]:
#                         outputs[i][j] = []
#                     if filter(output):
#                         outputs[i][j].append(output)
#         return outputs
#
#     def __notebook_run(
#         self,
#         version: int = DEFAULT_VERSION,
#         kernel: str = DEFAULT_KERNEL,
#         timeout: int = DEFAULT_TIMEOUT,
#     ):
#         if self.__notebook is not None:
#             return
#         with open(self.path, "r", encoding="utf-8") as f:
#             nb = nbformat.read(f, as_version=version)
#             ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)
#             ep.preprocess(nb, {"metadata": {"path": DIR_EXAMPLES}})
#             self.__notebook = nb
#
#     @property
#     def notebook(self):
#         self.__notebook_run()
#         return self.__notebook
#
#     @property
#     def errors_by_cell(self):
#         return self.__get_outputs(
#             self.notebook, lambda output: output.output_type == "error"
#         )
#
#     @property
#     def all_errors(self):
#         return [
#             error for _, outputs in self.errors_by_cell.items() for _, error in outputs
#         ]
#
#     @property
#     def warnings_by_cell(self):
#         return self.__get_outputs(
#             self.notebook, lambda output: output.output_type == "warning"
#         )
#
#     @property
#     def all_warnings(self):
#         return [
#             warning
#             for _, outputs in self.warnings_by_cell.items()
#             for _, warning in outputs
#         ]
#
#
# def fancy_name(path: Path) -> str:
#     names = path.with_suffix("").name.split("_")
#     if names[0].lower() in {"test", "demo"}:
#         names = names[1:]
#     return "".join(map(str.capitalize, names))
#
#
# class BaseNotebookTest(unittest.TestCase):
#     def setUp(self) -> None:
#         self.nbr = NotebookRunner(self.notebook_path)  # type: ignore
#         self.assertIsNotNone(self.nbr.notebook)
#
#     def test_notebook_has_no_errors(self):
#         for cell_index, outputs in self.nbr.errors_by_cell.items():
#             for output_index, output in outputs.items():
#                 with self.subTest(cell=cell_index, output=output_index):
#                     for error in output:
#                         self.fail(f"Error: {error}")
#
#
# for notebook in DIR_EXAMPLES.glob("*.ipynb"):
#     name = f"TestDemo{fancy_name(notebook)}"
#     klass = type(name, (BaseNotebookTest,), {"notebook_path": notebook})
#     globals()[name] = klass
#     print("Generate test case: ", name)
#
#
# del BaseNotebookTest
#
# if __name__ == "__main__":
#     unittest.main()
