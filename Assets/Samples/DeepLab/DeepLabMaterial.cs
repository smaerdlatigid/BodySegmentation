using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using System.Threading;
using TensorFlowLite;

public class DeepLabMaterial : MonoBehaviour
{
    private Thread deepLabThread;

    [SerializeField] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] Material mat = null;
    [SerializeField] ComputeShader compute = null;

    Texture2D camTexture;
    Texture2D deeplabTexture;
    WebCamTexture webcamTexture;
    DeepLab deepLab;

    [SerializeField] float maskPeriod = 0.2f;
    float nextInference = 0f;
    private Thread deeplabThread;

    void Start()
    {
        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
        webcamTexture.Play();
        camTexture = WebCamTexture2D(webcamTexture);

        // ML 
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        deepLab = new DeepLab(path, compute);
        deeplabTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.ARGB32, false);
        

        var resizeOptions = deepLab.ResizeOptions;
        resizeOptions.rotationDegree = webcamTexture.videoRotationAngle;
        deepLab.ResizeOptions = resizeOptions;

        //startThred();
    }

    void OnDestroy()
    {
        deepLabThread.Abort();
        webcamTexture?.Stop();
        deepLab?.Dispose();
    }

    Texture2D WebCamTexture2D(WebCamTexture Tex)
    {
        Texture2D tex = new Texture2D(Tex.width, Tex.height, TextureFormat.RGB24, false);
        var pixels = Tex.GetPixels(0, 0, Tex.width, Tex.height);
        tex.SetPixels(pixels);
        //tex.Compress(false);
        tex.Apply();
        return tex;
    }


    void startThred()
    {
    // https://docs.microsoft.com/en-us/dotnet/api/system.threading.thread?view=netframework-4.8
    // Start TcpServer background thread
        deeplabThread = new Thread(new ThreadStart(Infer));
        deeplabThread.IsBackground = true;
        deeplabThread.Start();
    }

    private void Infer()
    {
        while (true)
        {
            
            Debug.Log("infer");
        }
    }

    
    void Update()
    {
        deepLab.Invoke(webcamTexture);
        mat.SetTexture("Webcam",webcamTexture);
        deeplabTexture = deepLab.GetResultTexture2D();
        mat.SetTexture("Mask",deeplabTexture );
    }
}
