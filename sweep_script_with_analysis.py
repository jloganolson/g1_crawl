import subprocess
import itertools
from sweep_analyzer import analyze_sweep_results
import os
from datetime import datetime
import nanoid as nano




# Experiment configuration
EXPERIMENT_NAME = "g1_crawl_sweep_v13"  
START_FROM_RUN = 1  # Set to 1 to start from beginning, or higher to resume from a specific run

# =============================================================================
# PARAMETER SWEEP CONFIGURATION - CENTRALIZED
# =============================================================================
# Define which parameters to sweep (as lists) and which to keep fixed (as single values)
SWEEP_CONFIG = {
    # Parameters to sweep - each should be a list of values to test
    "SWEEP_PARAMS": {
        "env.rewards.anim_pose_l1.weight": [-2., -1., 0.],
        "env.rewards.anim_contact_mismatch_l1.weight": [-2., -1., 0.],
        "env.rewards.anim_forward_vel.weight": [5., 2., 1., 0.],
    },

}
# =============================================================================

# Exclude rules for parameter combinations.
# Each rule is a dict mapping parameter names to either:
#   - a single value (exact match), or
#   - a list/tuple/set of allowed values (membership match), or
#   - a callable predicate taking the candidate value and returning True/False.
# A combination is excluded if it matches ALL key conditions in ANY rule below.
EXCLUDE_CONFIG = [
    # Example: exclude the case where all three sweep params are 0.0
    {
        "env.rewards.anim_pose_l1.weight": 0.0,
        "env.rewards.anim_contact_mismatch_l1.weight": 0.0,
        "env.rewards.anim_forward_vel.weight": 0.0,
    },
]

def log_parameter_combination(combination, run_number, total_runs, log_file, status="STARTED", error=None, command_type=None, full_command=None, output_log_path=None):
    """Log parameter combination to a text file with status tracking."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\nRun #{run_number}/{total_runs} - {timestamp} - {status}")
        if command_type:
            f.write(f" ({command_type})")
        f.write("\n")
        for param_name, param_value in combination.items():
            f.write(f"{param_name}: {param_value}\n")
        if full_command:
            f.write(f"Command: {' '.join(full_command)}\n")
        if output_log_path:
            f.write(f"Output Log: {output_log_path}\n")
        if error:
            f.write(f"ERROR: {error}\n")
        f.write("-" * 50 + "\n")

def _value_matches_rule(candidate_value, rule_value):
    """Return True if candidate_value satisfies rule_value condition."""
    # Callable predicate
    if callable(rule_value):
        return bool(rule_value(candidate_value))
    # Membership over an iterable of options
    if isinstance(rule_value, (list, tuple, set)):
        return candidate_value in rule_value
    # Exact equality
    return candidate_value == rule_value


def _rule_matches_combination(combination, rule):
    """Return True if combination satisfies all key conditions in rule."""
    for param_name, rule_value in rule.items():
        # If the param is not part of this sweep combination, rule cannot match
        if param_name not in combination:
            return False
        if not _value_matches_rule(combination[param_name], rule_value):
            return False
    return True


def is_excluded_combination(combination):
    """Check EXCLUDE_CONFIG to decide if this combination should be filtered out."""
    if not EXCLUDE_CONFIG:
        return False
    for rule in EXCLUDE_CONFIG:
        if _rule_matches_combination(combination, rule):
            return True
    return False


def generate_parameter_combinations():
    """Generate all combinations of sweep parameters."""
    sweep_params = SWEEP_CONFIG["SWEEP_PARAMS"]
    
    if not sweep_params:
        return [{}]  # Return single empty combination if no sweep params
    
    # Get parameter names and their possible values
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    # Generate all combinations
    combinations = []
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        # Apply exclusion rules
        if is_excluded_combination(param_dict):
            continue
        combinations.append(param_dict)
    
    return combinations

def build_train_args(sweep_params):
    """Build training arguments from sweep and fixed parameters."""
    train_args = []
    
    # Add sweep parameters
    for param_name, param_value in sweep_params.items():
        train_args.append(f"{param_name}={param_value}")
    
   
    
    return train_args

def get_combination_description(sweep_params, combination_num, total_combinations):
    """Generate a description for the current parameter combination."""
    if not sweep_params:
        return f"Set {combination_num}/{total_combinations}: (fixed parameters only)"
    
    param_strs = [f"{param_name.split('.')[-1]}={param_value}" 
                  for param_name, param_value in sweep_params.items()]
    param_description = ", ".join(param_strs)
    
    return f"Set {combination_num}/{total_combinations}: {param_description}"

def run_command(command_args, description="Running command", prefix=None, log_file=None, run_number=None, total_runs=None, combination=None, command_type=None):
    """Helper function to execute shell commands and stream output to CLI."""
    print(f"\n--- {description} ---")
    print(f"Command: {' '.join(command_args)}")
    try:
        # Prepare per-run output capture file if logging is enabled
        output_log_path = None
        output_log_file = None
        if log_file and run_number and total_runs and command_type:
            sweep_dir = os.path.dirname(log_file)
            uuid_suffix = None
            if isinstance(combination, dict):
                uuid_suffix = combination.get("agent.run_name") or combination.get("run_name")
            safe_uuid = uuid_suffix if uuid_suffix else "NA"
            output_log_filename = f"{command_type.lower()}_run_{run_number:02d}_{safe_uuid}.log"
            output_log_path = os.path.join(sweep_dir, output_log_filename)
            # Write header for the output log
            with open(output_log_path, 'a') as outf:
                outf.write(f"=== {description} ===\n")
                outf.write(f"Command: {' '.join(command_args)}\n")
        
        # Popen allows streaming output
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream output line by line
        for line in process.stdout:
            if prefix:
                # Add prefix to each line, preserving original formatting
                formatted = f"[{prefix}] {line}"
                print(formatted, end='')
            else:
                formatted = line
                print(line, end='') # `end=''` prevents extra newlines
            # Append to output log if enabled
            if output_log_path:
                with open(output_log_path, 'a') as outf:
                    outf.write(formatted)

        process.wait() # Wait for the process to complete

        if process.returncode != 0:
            error_msg = f"Command exited with non-zero status {process.returncode}"
            print(f"\nError: {error_msg}")
            if log_file and run_number and total_runs and combination:
                log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg, command_type=command_type, full_command=command_args, output_log_path=output_log_path)
            return False
        else:
            print("\nCommand completed successfully.")
            if log_file and run_number and total_runs and combination:
                log_parameter_combination(combination, run_number, total_runs, log_file, status="COMPLETED", command_type=command_type, full_command=command_args, output_log_path=output_log_path)
            return True

    except FileNotFoundError:
        error_msg = f"Command not found. Make sure '{command_args[0]}' is in your PATH or correctly specified."
        print(f"Error: {error_msg}")
        if log_file and run_number and total_runs and combination:
            log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg, command_type=command_type, full_command=command_args, output_log_path=output_log_path if 'output_log_path' in locals() else None)
        return False
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(f"Error: {error_msg}")
        if log_file and run_number and total_runs and combination:
            log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg, command_type=command_type, full_command=command_args, output_log_path=output_log_path if 'output_log_path' in locals() else None)
        return False

def main():
    # Define base commands - updated to use EXPERIMENT_NAME
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--headless"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Create per-sweep output directory and log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_sweep_dir = "sweep-logs"
    os.makedirs(base_sweep_dir, exist_ok=True)
    sweep_output_dir = os.path.join(base_sweep_dir, f"{EXPERIMENT_NAME}_{timestamp}")
    os.makedirs(sweep_output_dir, exist_ok=True)
    log_file = os.path.join(sweep_output_dir, f"sweep_log_{EXPERIMENT_NAME}_{timestamp}.txt")
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Parameter Sweep Log - {EXPERIMENT_NAME}\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Resuming from run #{START_FROM_RUN}\n")
        f.write("\nSweep Configuration:\n")
        for param_name, param_values in SWEEP_CONFIG["SWEEP_PARAMS"].items():
            f.write(f"{param_name}: {param_values}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nRun Status Summary:\n")
        f.write("==================\n")

    # Generate parameter combinations
    parameter_combinations = generate_parameter_combinations()
    
    print(f"Starting parameter sweep: {EXPERIMENT_NAME}")
    print(f"Resuming from run #{START_FROM_RUN}")
    print(f"Generated {len(parameter_combinations)} parameter combinations to test.")
    print(f"Logging to: {log_file}")
    print(f"Sweep outputs directory: {sweep_output_dir}")
    
    # Print sweep configuration
    print(f"\nSweep parameters:")
    for param_name, param_values in SWEEP_CONFIG["SWEEP_PARAMS"].items():
        print(f"  {param_name}: {param_values}")
    
    for i, combination in enumerate(parameter_combinations):
        # Skip runs before START_FROM_RUN
        if i + 1 < START_FROM_RUN:
            continue
            
        description = get_combination_description(combination, i+1, len(parameter_combinations))
        run_prefix = f"Run {i+1}/{len(parameter_combinations)}"
        # Generate a short run identifier and augment combination for logging
        run_name = nano.generate(size=4)
        combo_with_names = dict(combination)
        combo_with_names["agent.experiment_name"] = EXPERIMENT_NAME
        combo_with_names["agent.run_name"] = run_name
        
        # Log the start of the parameter combination (includes UUID)
        log_parameter_combination(combo_with_names, i+1, len(parameter_combinations), log_file, status="STARTED")
        
        print(f"\n{'='*80}\nStarting {description}\n{'='*80}")

        # Construct parameter arguments
        train_args = build_train_args(combination)
        # Ensure experiment/run names are passed via Hydra overrides
        train_args += [
            f"agent.experiment_name={EXPERIMENT_NAME}",
            f"agent.run_name={run_name}",
        ]

        # Construct the full train command
        full_train_cmd = TRAIN_BASE_CMD + train_args
        train_ok = run_command(full_train_cmd, f"Training for {description}", prefix=run_prefix, 
                   log_file=log_file, run_number=i+1, total_runs=len(parameter_combinations), 
                   combination=combo_with_names, command_type="TRAIN")

        # Construct the full play command (no extra args needed for play usually)
        # Resolve the full run directory name (timestamp_UUID) for this run and pass to play
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", EXPERIMENT_NAME))
        matching_runs = []
        if os.path.isdir(log_root_path):
            for d in os.listdir(log_root_path):
                full_path = os.path.join(log_root_path, d)
                if os.path.isdir(full_path) and d.endswith(f"_{run_name}"):
                    matching_runs.append(d)
        if not matching_runs:
            # Fail loudly so it's clear what to fix
            raise RuntimeError(f"Could not find a run directory in '{log_root_path}' ending with '_{run_name}'")
        resolved_run_dir = sorted(matching_runs)[-1]
        combo_with_names["resolved_run_dir"] = resolved_run_dir

        play_args = [
            f"agent.experiment_name={EXPERIMENT_NAME}",
            f"agent.load_run={resolved_run_dir}",
        ]
        full_play_cmd = PLAY_BASE_CMD + play_args
        if not train_ok:
            # Skip play if training failed, but log it explicitly
            log_parameter_combination(combo_with_names, i+1, len(parameter_combinations), log_file, status="SKIPPED", command_type="PLAY", full_command=full_play_cmd)
            continue

        run_command(full_play_cmd, f"Playing for {description}", prefix=run_prefix,
                   log_file=log_file, run_number=i+1, total_runs=len(parameter_combinations),
                   combination=combo_with_names, command_type="PLAY")

    print(f"\n🎉 All {len(parameter_combinations)} parameter combinations finished!")
    
    # Automatically analyze results
    print(f"\n🔍 Starting automatic analysis of sweep results...")
    
    # Log analysis start
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Analysis Results\n")
        f.write("================\n")
        f.write(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Analysis with explicit output directory for all generated files
        results = analyze_sweep_results(experiment_name=EXPERIMENT_NAME, output_dir=sweep_output_dir)
        
        # Log analysis results
        with open(log_file, 'a') as f:
            if results and results['success']:
                f.write("\n✅ Analysis completed successfully!\n")
                f.write("Generated files:\n")
                f.write(f"  - Experiment guide: {results['experiment_guide_file']}\n")
                f.write(f"  - Basic concatenated video: {results['video_file']}\n")
                if results.get('labeled_video_file'):
                    f.write(f"  - Labeled concatenated video: {results['labeled_video_file']}\n")
                f.write(f"  - Video mapping: {results['video_mapping_file']}\n")
            else:
                f.write("\n⚠️ Analysis completed with issues\n")
                if results:
                    f.write(f"Error details: {results.get('error', 'Unknown error')}\n")
        
        if results and results['success']:
            print(f"\n🎊 SWEEP COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved:")
            print(f"   📄 Experiment guide: {results['experiment_guide_file']}")
            print(f"   🎬 Basic concatenated video: {results['video_file']}")
            if results.get('labeled_video_file'):
                print(f"   🏷️  Labeled concatenated video: {results['labeled_video_file']}")
            print(f"   🗺️  Video mapping: {results['video_mapping_file']}")
        else:
            print(f"\n⚠️  Sweep completed but analysis had issues. Check the output above.")
            
    except Exception as e:
        error_msg = f"Error during analysis: {e}"
        print(f"\n❌ {error_msg}")
        # Log analysis error
        with open(log_file, 'a') as f:
            f.write(f"\n❌ {error_msg}\n")
            f.write("You can manually run analysis later with:\n")
            f.write("python sweep_analyzer.py\n")
        print(f"   You can manually run analysis later with:")
        print(f"   python sweep_analyzer.py")

if __name__ == "__main__":
    main() 
    import time
    print("\n⏳ Waiting 5 seconds before suspend...")
    time.sleep(5)
    print("💤 Suspending computer...")
    subprocess.run(["systemctl", "suspend"], check=True)