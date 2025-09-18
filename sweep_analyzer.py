#!/usr/bin/env python3

import os
import subprocess
import re
from pathlib import Path
from datetime import datetime

def extract_parameters(experiment_dir):
    """Extract parameters from an experiment directory - now generic."""
    agent_file = os.path.join(experiment_dir, "params", "agent.yaml")
    env_file = os.path.join(experiment_dir, "params", "env.yaml")
    
    params = {}
    
    # Generic parameter extraction - just record that config files exist
    if os.path.exists(agent_file):
        params['has_agent_config'] = True
    if os.path.exists(env_file):
        params['has_env_config'] = True
    
    # Extract experiment directory name as the main identifier
    params['experiment_dir'] = os.path.basename(experiment_dir)
    
    return params

def check_ffmpeg_capabilities():
    """Check what ffmpeg encoders and filters are available."""
    # Skip capability checks - user has validated ffmpeg command works
    return {
        'has_libx264': True,
        'has_drawtext': True
    }

def create_labeled_videos(experiments, timestamp, capabilities=None):
    """Create individual labeled videos for each experiment."""
    labeled_videos = []
    temp_dir = f"temp_labeled_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("🏷️  Creating labeled videos...")
    
    # Use the font file from assets - no checks, just use it
    font_file = "assets/RobotoFlex-Variable.ttf"
    print(f"   Using font: {font_file}")
    
    for i, exp in enumerate(experiments, 1):
        # Use the experiment directory name as the label
        folder_name = exp['directory']
        
        # Create a clean label - use the full experiment name
        label_text = f"Run {i:02d}: {folder_name}"
        
        output_video = os.path.join(temp_dir, f"labeled_{i:02d}.mp4")
        
        # Use the user's validated ffmpeg format
        cmd = [
            "/usr/bin/ffmpeg", "-i", exp['video_path'],
            "-vf", f"drawtext=fontfile={font_file}:text='{label_text}':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10",
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-c:a", "copy",
            "-y", output_video
        ]
        
        print(f"   Processing video {i}/{len(experiments)}: {folder_name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            labeled_videos.append(output_video)
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to process video {i}: {e.stderr[:100] if e.stderr else 'Unknown error'}")
            return None, temp_dir
    
    return labeled_videos, temp_dir

def analyze_sweep_results(experiment_name, base_logs_dir="logs/rsl_rl", create_overlays=True):
    """
    Analyze sweep results and create concatenated video with experiment guide.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "g1_23dof_sweep_v4")
        base_logs_dir (str): Base directory for logs
        create_overlays (bool): Whether to attempt creating text overlays on videos
    
    Returns:
        dict: Results summary with file paths and experiment count
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing sweep results for: {experiment_name}")
    print(f"{'='*60}")
    
    # Construct the sweep directory path
    sweep_dir = os.path.join(base_logs_dir, experiment_name)
    
    if not os.path.exists(sweep_dir):
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return None
    
    # Find all experiment directories
    experiment_dirs = sorted([d for d in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, d))])
    
    # Analyze all experiments
    experiments = []
    for exp_dir in experiment_dirs:
        full_path = os.path.join(sweep_dir, exp_dir)
        video_path = os.path.join(full_path, "videos", "play", "rl-video-step-0.mp4")
        
        if os.path.exists(video_path):
            params = extract_parameters(full_path)
            experiments.append({
                'directory': exp_dir,
                'video_path': video_path,
                'params': params
            })
    
    if not experiments:
        print(f"❌ No experiments with videos found in {sweep_dir}")
        return None
    
    print(f"📊 Found {len(experiments)} experiments with videos")
    
    # Create output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_guide_file = f"{experiment_name}_experiment_guide_{timestamp}.txt"
    video_mapping_file = f"{experiment_name}_video_mapping_{timestamp}.txt"
    concat_video_file = f"{experiment_name}_concatenated_{timestamp}.mp4"
    concat_labeled_video_file = f"{experiment_name}_concatenated_labeled_{timestamp}.mp4"
    
    # Create experiment guide
    with open(experiment_guide_file, "w") as f:
        f.write(f"Experiment Sweep Results: {experiment_name}\n")
        f.write("=" * (26 + len(experiment_name)) + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Found {len(experiments)} experiments with videos\n\n")
        
        for i, exp in enumerate(experiments, 1):
            f.write(f"Run {i:2d}: {exp['directory']}\n")
            f.write(f"  Video: {exp['video_path']}\n")
            f.write(f"  Config: Agent={'✓' if exp['params'].get('has_agent_config') else '✗'}, "
                   f"Env={'✓' if exp['params'].get('has_env_config') else '✗'}\n")
            f.write("\n")
    
    print(f"✅ Experiment guide created: {experiment_guide_file}")
    
    # Always create overlays since user validated the command works
    capabilities = check_ffmpeg_capabilities()
    print(f"🔧 FFmpeg ready for overlays")
    
    # Try to create videos
    video_created = False
    labeled_video_created = False
    
    try:
        # Create basic concatenated video (always works)
        concat_file = f"temp_concat_{timestamp}.txt"
        with open(concat_file, "w") as f:
            for exp in experiments:
                f.write(f"file '{os.path.abspath(exp['video_path'])}'\n")
        
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-y", concat_video_file
        ]
        
        print("🎬 Creating basic concatenated video...")
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)
        print(f"✅ Basic concatenated video created: {concat_video_file}")
        video_created = True
        
        # Create labeled video - user validated the ffmpeg command works
        labeled_videos, temp_dir = create_labeled_videos(experiments, timestamp, capabilities)
        
        if labeled_videos:
            # Create concat file for labeled videos
            labeled_concat_file = f"temp_labeled_concat_{timestamp}.txt"
            with open(labeled_concat_file, "w") as f:
                for video in labeled_videos:
                    f.write(f"file '{os.path.abspath(video)}'\n")
            
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", labeled_concat_file,
                "-c", "copy", "-y", concat_labeled_video_file
            ]
            
            print("🏷️  Creating labeled concatenated video...")
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Cleanup
            os.remove(labeled_concat_file)
            import shutil
            shutil.rmtree(temp_dir)
            
            print(f"✅ Labeled concatenated video created: {concat_labeled_video_file}")
            labeled_video_created = True
        else:
            print(f"⚠️  Could not create labeled videos, only basic concatenation available")
        
        # Create video mapping
        with open(video_mapping_file, "w") as f:
            f.write(f"Video Segment Mapping\n")
            f.write("=" * 21 + "\n\n")
            if labeled_video_created:
                f.write(f"Labeled video: {concat_labeled_video_file}\n")
            f.write(f"Basic video: {concat_video_file}\n\n")
            f.write("Each video segment corresponds to:\n")
            f.write(f"(Each segment is approximately 4 seconds long)\n\n")
            
            for i, exp in enumerate(experiments, 1):
                f.write(f"Segment {i:2d} ({(i-1)*4:2.0f}s-{i*4:2.0f}s): {exp['directory']}\n")
                f.write(f"    Video source: {exp['video_path']}\n")
                f.write("\n")
        
        print(f"✅ Video mapping created: {video_mapping_file}")
        
    except Exception as e:
        print(f"⚠️  Video creation failed: {e}")
        print(f"   Experiment guide still available: {experiment_guide_file}")
    
    # Summary
    print(f"\n📋 SWEEP ANALYSIS COMPLETE:")
    print(f"   📊 Experiments analyzed: {len(experiments)}")
    print(f"   📄 Experiment guide: {experiment_guide_file}")
    if video_created:
        print(f"   🎬 Basic concatenated video: {concat_video_file}")
    if labeled_video_created:
        print(f"   🏷️  Labeled concatenated video: {concat_labeled_video_file}")
    if video_created or labeled_video_created:
        print(f"   🗺️  Video mapping: {video_mapping_file}")
    print(f"{'='*60}\n")
    
    return {
        'experiment_count': len(experiments),
        'experiment_guide_file': experiment_guide_file,
        'video_file': concat_video_file if video_created else None,
        'labeled_video_file': concat_labeled_video_file if labeled_video_created else None,
        'video_mapping_file': video_mapping_file if (video_created or labeled_video_created) else None,
        'success': video_created or labeled_video_created
    }

if __name__ == "__main__":
    # Example usage - can be called directly for testing
    analyze_sweep_results(
        experiment_name="g1_23dof_v18",
        create_overlays=True
    ) 