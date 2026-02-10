from tabulate import tabulate
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


metadata_rows = [
    [data.model, data.release, data.repo_id, len(data.saes_map)]
    for data in get_pretrained_saes_directory().values()
]

# Print all SAE releases, sorted by base model
print(
    tabulate(
        sorted(metadata_rows, key=lambda x: x[0]),
        headers=["model", "release", "repo_id", "n_saes"],
        tablefmt="simple_outline",
    )
)


release_id = input("enter the release:")

def format_value(value):
    return (
        "{{{0!r}: {1!r}, ...}}".format(*next(iter(value.items())))
        if isinstance(value, dict)
        else repr(value)
    )


release = get_pretrained_saes_directory()[release_id]

print(
    tabulate(
        [[k, format_value(v)] for k, v in release.__dict__.items()],
        headers=["Field", "Value"],
        tablefmt="simple_outline",
    )
)

data = [[id, path, release.neuronpedia_id[id]] for id, path in release.saes_map.items()]

print(
    tabulate(
        data,
        headers=["SAE id", "SAE path (HuggingFace)", "Neuronpedia ID"],
        tablefmt="simple_outline",
    )
)