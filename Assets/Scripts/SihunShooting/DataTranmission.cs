using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;

public class DataUploader : MonoBehaviour
{
    // Reuse the HttpClient if possible (static or a singleton).
    private static readonly HttpClient httpClient = new HttpClient();

    /// <summary>
    /// Call this method to upload a file from the Magic Leap device
    /// to your edge server.
    /// </summary>
    /// <param name="localFilePath">Full path on Magic Leap (e.g. /Documents/MyData/myfile.txt)</param>
    /// <param name="uploadUrl">Your server endpoint URL (e.g. http://192.168.1.10:5000/upload )</param>
    public async Task UploadFileAsync(string localFilePath, string uploadUrl)
    {
                // 1. Confirm the file exists
        if (!File.Exists(localFilePath))
        {
            Debug.LogError($"File not found: {localFilePath}");
            return;
        }
        else
        {
            Debug.Log($"Found file: {localFilePath}, preparing upload...");
        }

        // 2. Read the file bytes
        byte[] fileBytes = File.ReadAllBytes(localFilePath);

        // 3. Prepare the multipart form data
        using (var content = new MultipartFormDataContent())
        {
            var fileContent = new ByteArrayContent(fileBytes);

            // Name: "file", Filename: the actual file name
            content.Add(fileContent, "file", Path.GetFileName(localFilePath));

            try
            {
                // 4. POST to your edge server
                HttpResponseMessage response = await httpClient.PostAsync(uploadUrl, content);

                // 5. Check the result
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

    public async void UploadData(string localFilePath,string serverUrl)
    {
        await UploadFileAsync(localFilePath, serverUrl);
    }
  
}
