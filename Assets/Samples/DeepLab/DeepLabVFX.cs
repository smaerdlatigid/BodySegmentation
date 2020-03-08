using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using UnityEngine.VFX;

public class DeepLabVFX : MonoBehaviour
{
    [SerializeField] string fileName = "deeplabv3_257_mv_gpu.tflite";
    [SerializeField] VisualEffect vis = null;
    [SerializeField] ComputeShader compute = null;

    WebCamTexture webcamTexture;
    DeepLab deepLab;

    [SerializeField] float maskPeriod = 0.2f;
    float nextInference = 0f;

    void Start()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        deepLab = new DeepLab(path, compute);

        // Init camera
        string cameraName = WebCamUtil.FindName();
        webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
        webcamTexture.Play();

        var resizeOptions = deepLab.ResizeOptions;
        resizeOptions.rotationDegree = webcamTexture.videoRotationAngle;
        deepLab.ResizeOptions = resizeOptions;
    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        deepLab?.Dispose();
    }

    void Update()
    {
        vis.SetTexture("Webcam", webcamTexture);

        if (Time.time > nextInference)
        {
            deepLab.Invoke(webcamTexture);
            vis.SetTexture("Mask", deepLab.GetResultTexture2D());
            nextInference = Time.time + maskPeriod;
        }
    }
}
