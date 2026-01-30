"""
CLI tool for MIDI humanization.

Usage:
    python -m src.audio.humanize.cli input.mid output.mid --style romantic
    python -m src.audio.humanize.cli score.krn output.mid --format kern
"""

import click
from pathlib import Path
import logging

from .engine import HumanizationEngine
from .presets import get_preset, PRESETS
from .config import HumanizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_midi', type=click.Path())
@click.option(
    '--style',
    type=click.Choice(list(PRESETS.keys())),
    default='balanced',
    help='Performance style preset'
)
@click.option(
    '--format',
    type=click.Choice(['midi', 'kern', 'musicxml']),
    default='kern',
    help='Input file format'
)
@click.option(
    '--randomize/--no-randomize',
    default=True,
    help='Randomize k values for variation'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility'
)
@click.option(
    '--version',
    type=int,
    default=0,
    help='Version number for metadata'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Print k values and processing info'
)
def main(
    input_file: str,
    output_midi: str,
    style: str,
    format: str,
    randomize: bool,
    seed: int,
    version: int,
    verbose: bool
):
    """
    Humanize a MIDI file or score with KTH performance rules.

    Examples:

        humanize song.krn humanized.mid --style romantic

        humanize score.krn output.mid --format kern --seed 42

        humanize input.mid output.mid --no-randomize
    """
    # Set log level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get config
    click.echo(f"Loading preset: {style}")
    config = get_preset(style)

    # Randomize if requested
    if randomize:
        click.echo(f"Randomizing k values (seed: {seed})")
        config = config.randomize(seed=seed)

    if verbose:
        click.echo(f"\nk values:")
        for key, value in config.to_dict().items():
            click.echo(f"  {key}: {value:.3f}")
        click.echo()

    # Create engine
    engine = HumanizationEngine(config, seed=seed)

    # Process
    input_path = Path(input_file)
    output_path = Path(output_midi)

    click.echo(f"Processing: {input_path}")

    try:
        midi_file, metadata = engine.humanize_from_score(
            input_path,
            output_path,
            format=format,
            version=version
        )

        click.echo(f"✓ Humanized MIDI: {output_path}")
        click.echo(f"✓ Metadata: {output_path.with_suffix('.json')}")

        if verbose:
            click.echo(f"\nMetadata:")
            click.echo(f"  Version: {metadata.version}")
            click.echo(f"  Seed: {metadata.seed}")
            click.echo(f"  Timestamp: {metadata.timestamp}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


if __name__ == '__main__':
    main()
