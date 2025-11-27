"""
AICoverGen NextGen - Main Entry Point
Backward compatible wrapper for webui.py
"""

import gradio as gr

# Handle both module and script execution
try:
    from .pipeline import generate_cover
    from . import config
except ImportError:
    from pipeline import generate_cover
    import config


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, 
                        index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, 
                        f0_method='rmvpe', crepe_hop_length=128, protect=0.33, 
                        pitch_change_all=0, reverb_rm_size=0.15, reverb_wet=0.2, 
                        reverb_dry=0.8, reverb_damping=0.7, output_format='mp3',
                        progress=gr.Progress()):
    """
    Main pipeline function - backward compatible with webui.py
    """
    try:
        # Create progress callback for webui
        def progress_callback(message, percent):
            if is_webui:
                progress(percent, desc=message)
            else:
                print(message)
        
        # Call the new modular pipeline
        cover_path = generate_cover(
            song_input=song_input,
            voice_model=voice_model,
            pitch_change=pitch_change,
            keep_files=keep_files,
            main_gain=main_gain,
            inst_gain=inst_gain,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            f0_method=f0_method,
            crepe_hop_length=crepe_hop_length,
            protect=protect,
            pitch_change_all=pitch_change_all,
            reverb_room_size=reverb_rm_size,
            reverb_wet=reverb_wet,
            reverb_dry=reverb_dry,
            reverb_damping=reverb_damping,
            output_format=output_format,
            progress_callback=progress_callback
        )
        
        return cover_path
        
    except Exception as e:
        if is_webui:
            raise gr.Error(str(e))
        else:
            raise


# CLI support
if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate AI cover song')
    parser.add_argument('-i', '--song-input', type=str, required=True)
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True)
    parser.add_argument('-p', '--pitch-change', type=int, required=True)
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction)
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5)
    parser.add_argument('-fr', '--filter-radius', type=int, default=3)
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25)
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128)
    parser.add_argument('-pro', '--protect', type=float, default=0.33)
    parser.add_argument('-mv', '--main-vol', type=int, default=0)
    parser.add_argument('-iv', '--inst-vol', type=int, default=0)
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0)
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15)
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2)
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8)
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7)
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3')
    args = parser.parse_args()

    # Validate model exists
    model_path = config.RVC_MODELS_DIR / args.rvc_dirname
    if not model_path.exists():
        raise Exception(f'Model folder {model_path} does not exist.')

    cover_path = song_cover_pipeline(
        args.song_input, args.rvc_dirname, args.pitch_change, args.keep_files,
        main_gain=args.main_vol, inst_gain=args.inst_vol,
        index_rate=args.index_rate, filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate, f0_method=args.pitch_detection_algo,
        crepe_hop_length=args.crepe_hop_length, protect=args.protect,
        pitch_change_all=args.pitch_change_all,
        reverb_rm_size=args.reverb_size, reverb_wet=args.reverb_wetness,
        reverb_dry=args.reverb_dryness, reverb_damping=args.reverb_damping,
        output_format=args.output_format
    )
    print(f'[+] Cover generated at {cover_path}')
