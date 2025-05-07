import os
import glob
import click
import nbformat
from nbconvert import HTMLExporter, PDFExporter
import logging


def convert_notebook(filepath, output_dir, to_format):
    # Read the notebook
    with open(filepath, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Choose the appropriate exporter
    if to_format == 'html':
        exporter = HTMLExporter()
        ext = 'html'
    elif to_format == 'pdf':
        exporter = PDFExporter()
        ext = 'pdf'
    else:
        click.echo(f"Unsupported format: {to_format}")
        return

    # Convert the notebook
    converted, _ = exporter.from_notebook_node(notebook)

    # Prepare output file path
    base = os.path.splitext(os.path.basename(filepath))[0]
    # os.mkdir(output_dir, exist_ok=True, mode=0o755)
    output_file = os.path.join(output_dir, f"{base}.{ext}")

    # Write the converted file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(converted)

    click.echo(f"Converted '{filepath}' to '{output_file}'")

@click.command()
@click.option('--to', type=click.Choice(['html', 'pdf']), required=True, help="Output format: html or pdf.")
@click.option('--file', 'file_path', type=click.Path(exists=True), help="Path to a specific notebook file.")
@click.option('--pattern', type=str, help="Glob pattern to search for notebook files.")
@click.option('--output-dir', type=click.Path(file_okay=False), help="Directory to store the output files when using pattern.")
def main(to, file_path, pattern, output_dir):
    """
    Convert Jupyter Notebooks to HTML or PDF using nbconvert.
    """
    if file_path and pattern:
        click.echo("Error: Please specify either --file or --pattern, not both.")
        return

    if not file_path and not pattern:
        click.echo("Error: Please specify either --file or --pattern.")
        return

    if file_path:
        # Single file conversion; output directory defaults to the file's directory.
        output_dir_local = os.path.dirname(file_path)
        convert_notebook(file_path, output_dir_local, to)
    else:
        # Batch conversion using glob pattern.
        files = glob.glob(pattern, recursive=True)
        if not files:
            click.echo("Error: No files found matching the given pattern.")
            click.echo(f"{file_path=} is not a valid notebook file.")
            return

        if not output_dir:
            click.echo("Error: Please provide --output-dir for pattern conversion.")
            return

        # Create the output directory if it does not exist.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file in files:
            try:
                convert_notebook(file, output_dir, to)
            except Exception as e:
                click.echo(f"Error converting '{file}': {e}")
                logging.error(f"Error converting '{file}': {e}")
                continue
if __name__ == '__main__':
    main()