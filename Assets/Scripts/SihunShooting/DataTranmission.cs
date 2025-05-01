using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class DataUploader : MonoBehaviour
{
    private static readonly HttpClient httpClient = new HttpClient();
    private string serverIpAddress = null;

    /// <summary>
    /// Listen for the server's signal to get its IP address.
    /// </summary>
    public async Task ListenForServerSignalAsync()
    {
        using (UdpClient udpClient = new UdpClient(5001))
        {
            try
            {
                Debug.Log("Listening for server signal...");
                var receiveTask = udpClient.ReceiveAsync();
                if (await Task.WhenAny(receiveTask, Task.Delay(20000)) == receiveTask)
                {
                    UdpReceiveResult result = receiveTask.Result;
                    serverIpAddress = Encoding.UTF8.GetString(result.Buffer);
                    Debug.Log($"Received server IP address: {serverIpAddress}");

                    // Send acknowledgment ("ACK") back to the server
                    using (UdpClient ackClient = new UdpClient())
                    {
                        byte[] ackMessage = Encoding.UTF8.GetBytes("ACK");
                        await ackClient.SendAsync(ackMessage, ackMessage.Length, result.RemoteEndPoint);
                        Debug.Log("Acknowledgment sent to the server.");
                    }
                }
                else
                {
                    Debug.LogError("Listening for server signal timed out.");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Exception while listening for server signal: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Upload a file to the server.
    /// </summary>
    public async Task UploadFileAsync(string localFilePath)
    {
        if (string.IsNullOrEmpty(serverIpAddress))
        {
            await ListenForServerSignalAsync();
        }

        if (string.IsNullOrEmpty(serverIpAddress))
        {
            Debug.LogError("Server IP address is not available. Cannot upload the file.");
            return;
        }

        string uploadUrl = $"http://{serverIpAddress}:5000/upload";

        if (!File.Exists(localFilePath))
        {
            Debug.LogError($"File not found: {localFilePath}");
            return;
        }

        byte[] fileBytes = File.ReadAllBytes(localFilePath);

        using (var content = new MultipartFormDataContent())
        {
            var fileContent = new ByteArrayContent(fileBytes);
            content.Add(fileContent, "file", Path.GetFileName(localFilePath));

            try
            {
                HttpResponseMessage response = await httpClient.PostAsync(uploadUrl, content);
                if (response.IsSuccessStatusCode)
                {
                    Debug.Log("File uploaded successfully!");
                }
                else
                {
                    Debug.LogError($"Upload failed with HTTP status {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Upload exceptions: {ex.Message}");
            }
        }
    }

    public async void UploadData(string localFilePath)
    {
        await UploadFileAsync(localFilePath);
    }
}
