using System.IO;
using GaussianSplatting.Runtime;
using Newtonsoft.Json;
using UnityEngine;

public class GaussianSplatTester : MonoBehaviour
{
    private GaussianSplatRenderer _renderer;

    private void Start()
    {
        _renderer = GetComponent<GaussianSplatRenderer>();
    }

    private void Update()
    {
        if (enabled && Input.GetKeyDown(KeyCode.L))
        {
            var stream = File.OpenRead("C:\\Repositories\\Sonnberg2\\Sonnberg2.splat");
            using (var sr = new StreamReader(stream))
            using (var jsonTextReader = new JsonTextReader(sr))
            {
                var splat = GaussianUtils.GetSplatSerializer().Deserialize<GaussianSplat>(jsonTextReader);
                splat.SetSplatFiles();
                _renderer.Splat = splat;
            }
        }
    }
}
