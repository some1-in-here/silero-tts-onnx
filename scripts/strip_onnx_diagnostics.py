"""Remove exporter diagnostics, never weights/operators, from release graphs."""
import re

import onnx


def strip_diagnostics(path):
    model = onnx.load(str(path))
    removed = 0
    stack = [model]
    while stack:
        message = stack.pop()
        fields = message.DESCRIPTOR.fields_by_name
        if "doc_string" in fields and message.doc_string:
            message.ClearField("doc_string")
            removed += 1
        if "metadata_props" in fields and message.metadata_props:
            removed += len(message.metadata_props)
            message.ClearField("metadata_props")
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                stack.extend(value if field.is_repeated else [value])
            elif field.type == field.TYPE_STRING:
                values = value if field.is_repeated else [value]
                for text in values:
                    if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/", text):
                        raise ValueError("A local path remains outside diagnostic metadata")
    # Only protobuf diagnostic fields above are cleared. Initializers,
    # operators, names, shapes, attributes and opset imports are untouched.
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return removed
