namespace MagicLeap.Examples
{
    using MagicLeap.Core;
    using SimpleJson;
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using UnityEngine;
    using UnityEngine.Networking;
    using UnityEngine.SceneManagement;
    using UnityEngine.UI;
    using UnityEngine.XR.MagicLeap;

    // Disabling WebRTC deprecated warning for the examples project
    #pragma warning disable 618
    public class MLWebRTCExample : MonoBehaviour
    {
        private MLWebRTC.PeerConnection connection = null;
        private MLWebRTC.MLCameraVideoSource localVideoSource;
        private MLWebRTC.MediaStream localMediaStream = null;

        private MLCamera mlCamera;
        private MLCamera.ConnectContext connectContext;
        private string serverAddress = "";
        private string serverURI = "";
        private int captureWidth = 648;
        private int captureHeight = 720;

        private readonly MLPermissions.Callbacks permissionCallbacks = new MLPermissions.Callbacks();
        private static readonly string[] requiredPermissions = new string[] { MLPermission.Camera };
        private readonly HashSet<string> grantedPermissions = new HashSet<string>();

        private void Awake()
        {
            permissionCallbacks.OnPermissionGranted += OnPermissionGranted;
            permissionCallbacks.OnPermissionDenied += OnPermissionDenied;
            permissionCallbacks.OnPermissionDeniedAndDontAskAgain += OnPermissionDenied;
        }

        private void Start()
        {
            foreach (string permission in requiredPermissions)
            {
                MLPermissions.RequestPermission(permission, permissionCallbacks);
            }
        }

        private void OnPermissionGranted(string permission)
        {
            grantedPermissions.Add(permission);
            if (grantedPermissions.Count == requiredPermissions.Length)
            {
                StartAfterPermissions();
            }
        }

        private void OnPermissionDenied(string permission)
        {
            Debug.LogError($"Permission denied: {permission}. Application cannot proceed.");
        }

        private void StartAfterPermissions()
        {
            Debug.Log("Permissions granted. Setting up WebRTC...");
            ConnectToServer("192.168.1.23"); // Replace with your server address
        }

        private void ConnectToServer(string address)
        {
            serverAddress = address;
            serverURI = $"http://{serverAddress}:8080";
            Debug.Log($"Connecting to server at {serverURI}...");
            InitializeWebRTCConnection();
        }

        private async void InitializeWebRTCConnection()
        {
            connection = MLWebRTC.PeerConnection.CreateRemote(CreateIceServers(), out MLResult result);
            if (!result.IsOk)
            {
                Debug.LogError($"Failed to create WebRTC connection. Reason: {MLResult.CodeToString(result.Result)}");
                return;
            }

            SubscribeToConnection(connection);
            await CreateLocalMediaStream();
            InitTracks();
        }

        private async Task CreateLocalMediaStream()
        {
            Debug.Log("Setting up local media stream...");

            connectContext = new MLCamera.ConnectContext()
            {
                CamId = MLCamera.Identifier.Main,
                Flags = MLCamera.ConnectFlag.CamOnly,
                EnableVideoStabilization = true
            };

            mlCamera = await MLCamera.CreateAndConnectAsync(connectContext);
            mlCamera.OnRawImageAvailable+=OnCaptureRawVideoFrameAvailable;
            if (mlCamera == null)
            {
                Debug.LogError("Failed to create and connect to the camera.");
                return;
            }

            MLCamera.StreamCapability[] streamCapabilities = MLCamera.GetImageStreamCapabilitiesForCamera(mlCamera, MLCamera.CaptureType.Video);
            if (streamCapabilities.Length == 0)
            {
                Debug.LogError("No stream capabilities found for the camera.");
                return;
            }

            MLCamera.CaptureConfig captureConfig = new MLCamera.CaptureConfig()
            {
                CaptureFrameRate = MLCamera.CaptureFrameRate._30FPS,
                StreamConfigs = new[] { MLCamera.CaptureStreamConfig.Create(streamCapabilities[0], MLCamera.OutputFormat.YUV_420_888) }
            };

            localVideoSource = MLWebRTC.MLCameraVideoSource.CreateLocal(mlCamera, captureConfig, out MLResult result, "localStream", null, false);
            if (localVideoSource == null || !result.IsOk)
            {
                Debug.LogError("Failed to create local video source.");
                return;
            }

            localMediaStream = MLWebRTC.MediaStream.CreateWithAppDefinedVideoTrack("localStream", localVideoSource, MLWebRTC.MediaStream.Track.AudioType.None, "", null);
            Debug.Log("Local media stream created.");
        }
        private void OnCaptureRawVideoFrameAvailable(MLCamera.CameraOutput capturedFrame, MLCamera.ResultExtras resultExtras, MLCamera.Metadata metadataHandle)
        {
            Debug.Log("raw video frame callback");

        }
        
        private void InitTracks()
        {
            if (connection == null || localMediaStream == null)
            {
                Debug.LogError("Cannot initialize tracks - connection or local media stream is null.");
                return;
            }

            connection.AddLocalTrack(localMediaStream.ActiveVideoTrack);
            Debug.Log("Local video track added to the WebRTC connection.");
        }

        private MLWebRTC.IceServer[] CreateIceServers()
        {
            return new[]
            {
                MLWebRTC.IceServer.Create("stun:stun.l.google.com:19302")
            };
        }

        private void SubscribeToConnection(MLWebRTC.PeerConnection connection)
        {
            connection.OnError += (conn, error) => Debug.LogError($"WebRTC Connection Error: {error}");
            connection.OnConnected += (conn) => Debug.Log("WebRTC Connection established.");
            connection.OnDisconnected += (conn) => Debug.Log("WebRTC Connection disconnected.");
        }
    }
}
