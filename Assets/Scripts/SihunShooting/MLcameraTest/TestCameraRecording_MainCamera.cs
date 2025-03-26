using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MagicLeap.Core;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.MagicLeap;

namespace MagicLeap.Examples
{
    public class TestCameraRecording_MainCamera : MonoBehaviour
    {
        private MLCamera.OutputFormat OutputFormat = MLCamera.OutputFormat.RGBA_8888;
        private MLCamera captureCamera;
        public bool isCapturingVideo = false;
        
        private int _targetImageWidth = 1440;
        private int _targetImageHeight = 1080;
        public MLCameraBase.CaptureFrameRate _targetFrameRate = MLCameraBase.CaptureFrameRate._60FPS;

        private readonly CameraRecorder cameraRecorder = new CameraRecorder();

        private const string validFileFormat = "mp4";
        private string baseFileName = "egocentric_vdieo";

        private string recordedFilePath;
        private MLCamera.CaptureType CaptureType = MLCamera.CaptureType.Video;

        private List<MLCamera.StreamCapability> streamCapabilities;
        private readonly MLPermissions.Callbacks permissionCallbacks = new MLPermissions.Callbacks();

        private bool cameraDeviceAvailable;


        private void Awake()
        {
            permissionCallbacks.OnPermissionGranted += OnPermissionGranted;
            permissionCallbacks.OnPermissionDenied += OnPermissionDenied;
            permissionCallbacks.OnPermissionDeniedAndDontAskAgain += OnPermissionDenied;
        }

        private void Start()
        {
            Debug.Log("Start");
            MLPermissions.RequestPermission(MLPermission.Camera, permissionCallbacks);
            MLPermissions.RequestPermission(MLPermission.RecordAudio, permissionCallbacks);
            TryEnableMLCamera();
        }

        private void TryEnableMLCamera()
        {
            if (!MLPermissions.CheckPermission(MLPermission.Camera).IsOk)
                return;

            StartCoroutine(EnableMLCamera());
        }

        private IEnumerator EnableMLCamera()
        {
            while (!cameraDeviceAvailable)
            {
                MLResult result =
                    MLCamera.GetDeviceAvailabilityStatus(MLCamera.Identifier.Main, out cameraDeviceAvailable);
                if (!(result.IsOk && cameraDeviceAvailable))
                {
                    // Wait until camera device is available
                    yield return new WaitForSeconds(0.5f);
                }
                else
                {
                    ConnectCamera();
                }
            }
            Debug.Log("Camera device available");
        }

        private void Update()
        {
            // Debug.Log($"Frame Number");

        }

        private void OnPermissionDenied(string permission)
        {
            if (permission == MLPermission.Camera)
            {
                MLPluginLog.Error($"{permission} denied, example won't function.");
            }
            else if (permission == MLPermission.RecordAudio)
            {
                MLPluginLog.Error($"{permission} denied, audio wont be recorded in the file.");
            }
        }

        private void OnPermissionGranted(string permission)
        {
            MLPluginLog.Debug($"Granted {permission}.");
            // TryEnableMLCamera();
        }

        public void StartVideoCapture(int taskindex)
        {
            // TryEnableMLCamera();

            var result = MLPermissions.CheckPermission(MLPermission.Camera);
            MLResult.DidNativeCallSucceed(result.Result, nameof(MLPermissions.RequestPermission));
            Debug.Log($"CLPermissions.CheckPermission {result}");
            if (!result.IsOk)
            {
                Debug.LogError($"{MLPermission.Camera} permission denied. Video will not be recorded.");
                return;
            }

            StartRecording(taskindex);

            }

       private void StartRecording(int taskindex)
        {

            string fileName = SharedInfomanager.Instance.GenerateUniqueFilePath(baseFileName, taskindex, validFileFormat);

            recordedFilePath = System.IO.Path.Combine(Application.persistentDataPath, fileName);

            CameraRecorderConfig config = CameraRecorderConfig.CreateDefault();
            config.Width = _targetImageWidth;
            config.Height = _targetImageHeight;
            config.FrameRate = _targetFrameRate == MLCameraBase.CaptureFrameRate._60FPS ? 60 : 30;

            cameraRecorder.StartRecording(recordedFilePath, config);
            // Subscribe to the OnInfo event
            
            isCapturingVideo=true;

            MLCamera.CaptureConfig captureConfig = new MLCamera.CaptureConfig();
            captureConfig.CaptureFrameRate = _targetFrameRate;
            captureConfig.StreamConfigs = new MLCamera.CaptureStreamConfig[1];
            captureConfig.StreamConfigs[0] = MLCamera.CaptureStreamConfig.Create(streamCapabilities[0], OutputFormat);
            captureConfig.StreamConfigs[0].Surface = cameraRecorder.MediaRecorder.InputSurface;


            MLResult result = captureCamera.PrepareCapture(captureConfig, out MLCamera.Metadata _);

            if (MLResult.DidNativeCallSucceed(result.Result, nameof(captureCamera.PrepareCapture)))
            {
                captureCamera.PreCaptureAEAWB();

                if (CaptureType == MLCamera.CaptureType.Video)
                {
                    result = captureCamera.CaptureVideoStart();
                    // Debug.Log($"Video recording started successfully. at {Time.time}");

                    isCapturingVideo = MLResult.DidNativeCallSucceed(result.Result, nameof(captureCamera.CaptureVideoStart));
                    // SharedInfomanager.Instance.SetStartrecordingtime(Time.time);
                    SharedInfomanager.Instance.startVideo = 1;
                    SharedInfomanager.Instance.SetStartrecordingtime(Time.time);

                    if (isCapturingVideo)
                    {
                        Debug.Log($"Video recording started successfully. at {Time.time}");
                    }
                }
            }
        }




        public void StopRecording()
        {

            captureCamera.CaptureVideoStop();            
            Debug.LogWarning("CaptureVideoStop done.");


            if (!isCapturingVideo)
            {
                Debug.LogWarning("No recording is in progress to stop.");
                return;
            }

            SharedInfomanager.Instance.startVideo = 3;
            SharedInfomanager.Instance.SetEndrecordingtime(Time.time);

            MLResult result = cameraRecorder.EndRecording();

            if (!result.IsOk)
            {
                Debug.LogError($"Failed to stop recording: {result}");
            } 
            isCapturingVideo = false;

        }
    
        private void ConnectCamera()
        {
            MLCamera.ConnectContext context = MLCamera.ConnectContext.Create();
            context.Flags = MLCamera.ConnectFlag.MR;
            // context.EnableVideoStabilization = true;

            if (context.Flags != MLCamera.ConnectFlag.CamOnly)
            {
                context.MixedRealityConnectInfo = MLCamera.MRConnectInfo.Create();
                context.MixedRealityConnectInfo.MRQuality =MLCameraBase.MRQuality._1440x1080;
                context.MixedRealityConnectInfo.MRBlendType = MLCamera.MRBlendType.Additive;
                context.MixedRealityConnectInfo.FrameRate = _targetFrameRate;
            }

            captureCamera = MLCamera.CreateAndConnect(context);

            if (captureCamera != null)
            {

                Debug.Log("Camera device connected");
                if (GetImageStreamCapabilities())
                {
                    Debug.Log("Camera stream capabilities received.");
                }
            }
        }
        private void DisconnectCamera()
        {
            if (captureCamera != null)
            {
                captureCamera.Disconnect();
                // MLCamera.Uninitialize();

                captureCamera = null;
                Debug.Log("Camera disconnected");
            }
        }
        private bool GetImageStreamCapabilities()
        {
            var result =
                captureCamera.GetStreamCapabilities(out MLCamera.StreamCapabilitiesInfo[] streamCapabilitiesInfo);

            if (!result.IsOk)
            {
                Debug.LogError("Failed to get stream capabilities info.");
                return false;
            }

            streamCapabilities = new List<MLCamera.StreamCapability>();

            foreach (var info in streamCapabilitiesInfo)
            {
                streamCapabilities.AddRange(info.StreamCapabilities);
            }

            return streamCapabilities.Count > 0;
        }

        
    }
}
