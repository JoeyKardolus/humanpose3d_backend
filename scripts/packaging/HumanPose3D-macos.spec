# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for macOS builds
#
# Recommended usage (via build script):
#   ./scripts/packaging/build.sh macos
#
# Direct usage (requires flags):
#   uv run pyinstaller scripts/packaging/HumanPose3D-macos.spec -y --distpath bin --workpath bin/build

from pathlib import Path

# SPECPATH is the directory containing this spec file (scripts/packaging/)
# Go up two levels to get to repo root
repo_root = Path(SPECPATH).resolve().parent.parent

a = Analysis(
    [str(repo_root / 'scripts' / 'packaging' / 'pyinstaller_entry.py')],
    pathex=[str(repo_root)],
    binaries=[],
    datas=[
        (str(repo_root / 'src' / 'application' / 'templates'), 'src/application/templates'),
        (str(repo_root / 'src' / 'application' / 'static'), 'src/application/static'),
        (str(repo_root / 'models'), 'models'),
    ],
    hiddenimports=[
        'src.__init__',
        'src.application.__init__', 'src.application.urls', 'src.application.apps',
        'src.application.views', 'src.application.build_log',
        'src.datastream.__init__', 'src.datastream.data_stream',
        'src.datastream.marker_estimation', 'src.datastream.post_augmentation_estimation',
        'src.datastream.apps',
        'src.depth_refinement.__init__', 'src.depth_refinement.data_utils',
        'src.depth_refinement.dataset', 'src.depth_refinement.inference',
        'src.depth_refinement.losses', 'src.depth_refinement.model', 'src.depth_refinement.apps',
        'src.joint_refinement.__init__', 'src.joint_refinement.dataset',
        'src.joint_refinement.inference', 'src.joint_refinement.losses',
        'src.joint_refinement.model', 'src.joint_refinement.apps',
        'src.kinematics.__init__', 'src.kinematics.angle_processing',
        'src.kinematics.comprehensive_joint_angles', 'src.kinematics.segment_coordinate_systems',
        'src.kinematics.trc_utils', 'src.kinematics.visualize_comprehensive_angles',
        'src.kinematics.apps',
        'src.main_refinement.__init__', 'src.main_refinement.dataset',
        'src.main_refinement.inference', 'src.main_refinement.losses',
        'src.main_refinement.model', 'src.main_refinement.apps',
        'src.markeraugmentation.gpu_config', 'src.markeraugmentation.markeraugmentation',
        'src.markeraugmentation.apps', 'src.markeraugmentation.__init__',
        'src.mediastream.__init__', 'src.mediastream.media_stream', 'src.mediastream.apps',
        'src.pipeline.__init__', 'src.pipeline.cleanup', 'src.pipeline.refinement',
        'src.pipeline.runner', 'src.pipeline.apps',
        'src.posedetector.__init__', 'src.posedetector.pose_detector', 'src.posedetector.apps',
        'src.postprocessing.__init__', 'src.postprocessing.temporal_smoothing',
        'src.postprocessing.apps',
        'src.visualizedata.__init__', 'src.visualizedata.visualize_data', 'src.visualizedata.apps',
        'src.api.__init__', 'src.api.apps', 'src.api.urls', 'src.api.views',
        'src.cli.__init__', 'src.cli.apps',
        'src.application.config.__init__', 'src.application.config.paths',
        'src.application.controllers.__init__', 'src.application.controllers.pipeline_views',
        'src.application.dto.__init__', 'src.application.dto.pipeline_execution_result',
        'src.application.dto.pipeline_request', 'src.application.dto.pipeline_run_spec',
        'src.application.dto.progress_payload', 'src.application.dto.pipeline_preparation_result',
        'src.application.repositories.__init__', 'src.application.repositories.run_status_repository',
        'src.application.services.__init__', 'src.application.services.landmark_plot_service',
        'src.application.services.output_directory_service',
        'src.application.services.output_history_service',
        'src.application.services.pipeline_command_builder',
        'src.application.services.pipeline_log_service',
        'src.application.services.results_archive_service',
        'src.application.services.results_service', 'src.application.services.run_id_factory',
        'src.application.services.trc_plot_service', 'src.application.services.upload_service',
        'src.application.services.media_service', 'src.application.services.pipeline_progress_tracker',
        'src.application.services.pipeline_runner', 'src.application.services.progress_service',
        'src.application.services.run_cleanup_service', 'src.application.services.run_key_service',
        'src.application.services.statistics_service', 'src.application.services.pipeline_result_service',
        'src.application.use_cases.__init__', 'src.application.use_cases.prepare_pipeline_run',
        'src.application.use_cases.run_pipeline_sync', 'src.application.use_cases.run_pipeline_async',
        'src.application.validators.__init__', 'src.application.validators.path_validator',
        'src.application.validators.run_request_validator',
        'src.cli.management.__init__', 'src.cli.management.commands.run_pipeline',
        'src.cli.management.commands.__init__',
        'humanpose3d.__init__', 'humanpose3d.asgi', 'humanpose3d.wsgi',
        'humanpose3d.settings', 'humanpose3d.urls',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HumanPose3D-macos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HumanPose3D-macos',
)
