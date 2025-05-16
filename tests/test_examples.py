import nbformat
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor
import unittest
from typing import Callable


DIR_ROOT = Path(__file__).resolve().parent.parent
DIR_EXAMPLES = DIR_ROOT / "examples"


class NotebookRunner:
    DEFAULT_VERSION = 4
    DEFAULT_KERNEL = "python3"
    DEFAULT_TIMEOUT = 600

    def __init__(self, path: Path) -> None:
        self.path = path
        self.__notebook = None

    def __get_outputs(self, notebook, filter: Callable) -> dict[int, dict[int, list]]:
        outputs: dict[int, dict[int, list]] = {}
        for i, cell in enumerate(notebook.cells):
            if "outputs" in cell:
                for j, output in enumerate(cell["outputs"]):
                    if filter(output):
                        outputs.setdefault(i, {}).setdefault(j, []).append(output)
        return outputs

    def __notebook_run(
        self,
        version: int = DEFAULT_VERSION,
        kernel: str = DEFAULT_KERNEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        if self.__notebook is not None:
            return

        with open(self.path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=version)
            ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)

            try:
                ep.preprocess(nb, {"metadata": {"path": self.path.parent}})
            except Exception as e:
                print(f"\n❌ Error during execution of notebook: {self.path.name}")
                raise e

            self.__notebook = nb

    @property
    def notebook(self):
        self.__notebook_run()
        return self.__notebook

    @property
    def errors_by_cell(self):
        return self.__get_outputs(
            self.notebook, lambda output: output.output_type == "error"
        )

    @property
    def all_errors(self):
        return [
            error
            for _, outputs in self.errors_by_cell.items()
            for _, error in outputs.items()
            for error in error
        ]


def fancy_name(path: Path) -> str:
    stem = path.with_suffix("").name
    parts = stem.split("_")
    if parts[0].lower() in {"test", "demo"}:
        parts = parts[1:]
    return "".join(p.capitalize() for p in parts)


class BaseNotebookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nbr = NotebookRunner(self.notebook_path)  # type: ignore
        self.assertIsNotNone(self.nbr.notebook)

    def test_notebook_has_no_errors(self):
        for cell_index, outputs in self.nbr.errors_by_cell.items():
            for output_index, output in outputs.items():
                for error in output:
                    with self.subTest(cell=cell_index, output=output_index):
                        print(f"\n🚨 Error in cell {cell_index} output {output_index}:")
                        print("Traceback:")
                        if "traceback" in error:
                            print("".join(error["traceback"]))
                        self.fail(f"Notebook error in cell {cell_index}")


# Generazione dinamica dei test
existing_names = set()

import json

for notebook in DIR_EXAMPLES.glob("*.ipynb"):
    try:
        with open(notebook, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        # Check versione del notebook
        if "nbformat" not in json_data or json_data["nbformat"] < 4:
            print(f"⚠️  Skipping {notebook.name}: unsupported nbformat version")
            continue
    except Exception as e:
        print(f"⚠️  Skipping {notebook.name}: not a valid notebook file ({e})")
        continue

    base_name = fancy_name(notebook)
    class_name = f"TestDemo{base_name}"

    counter = 2
    while class_name in existing_names:
        class_name = f"TestDemo{base_name}{counter}"
        counter += 1

    existing_names.add(class_name)

    klass = type(class_name, (BaseNotebookTest,), {"notebook_path": notebook})
    globals()[class_name] = klass
    print(f"✅ Generated test case: {class_name} for notebook {notebook.name}")


# Cleanup per evitare esecuzione della base
del BaseNotebookTest

if __name__ == "__main__":
    unittest.main()
