import click
from .commands.gradcam import gradcam_cmd
from .gradcam.cam import CONTINUOUS_TARGETS, LAYER_SELECTIONS, METHODS

@click.group()
@click.pass_context
def cli(ctx):
    """
    Can be used to set global config options for all commands.
    """

    ctx.ensure_object(dict)

@cli.command()
@click.option("-i", "--input_mp4", required=True, help="The path to an MP4 input file.")
@click.option("-o", "--output_mp4", required=True, help="The path to an MP4 output file.")
@click.option("-m", "--model", required=True, help="The path to a model.pb file.")
@click.option("-mm", "--model_metadata", required=True, help="The path to a model metadata file.")
@click.option("--fps", type=int, default=15, help="The path to a model metadata file.")
@click.option(
    "--method",
    type=click.Choice(list(METHODS)),
    default="hirescam",
    show_default=True,
    help="Class activation mapping method.",
)
@click.option(
    "-a",
    "--action",
    type=int,
    default=None,
    help="Discrete action index to explain. Defaults to the action selected by the model in each frame.",
)
@click.option(
    "--continuous-target",
    type=click.Choice(list(CONTINUOUS_TARGETS)),
    default="steering",
    show_default=True,
    help="Output to explain for continuous action space models.",
)
@click.option(
    "--layers",
    type=click.Choice(list(LAYER_SELECTIONS)),
    default=None,
    help="Convolutional layers to explain. Defaults to 'all' for layercam and 'last' otherwise.",
)
@click.pass_context
def gradcam(ctx, input_mp4, output_mp4, model, model_metadata, fps, method, action, continuous_target, layers):
    """
    Process an input MP4 by applying gradcam to each frame of the video using the provided model.
    Outputs a video with a gradcam overlay.
    """

    gradcam_cmd(
        input_file=input_mp4, 
        output_file=output_mp4,
        model=model, 
        model_metadata=model_metadata, 
        fps=fps,
        method=method,
        action=action,
        continuous_target=continuous_target,
        layers=layers,
    )