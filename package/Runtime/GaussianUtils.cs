// SPDX-License-Identifier: MIT

using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    public static class GaussianUtils
    {
        public static float Sigmoid(float v)
        {
            return math.rcp(1.0f + math.exp(-v));
        }

        public static float3 SH0ToColor(float3 dc0)
        {
            const float kSH_C0 = 0.2820948f;
            return dc0 * kSH_C0 + 0.5f;
        }

        public static float3 LinearScale(float3 logScale)
        {
            return math.abs(math.exp(logScale));
        }

        public static float SquareCentered01(float x)
        {
            x -= 0.5f;
            x *= x * math.sign(x);
            return x * 2.0f + 0.5f;
        }

        public static float InvSquareCentered01(float x)
        {
            x -= 0.5f;
            x *= 0.5f;
            x = math.sqrt(math.abs(x)) * math.sign(x);
            return x + 0.5f;
        }

        public static float4 NormalizeSwizzleRotation(float4 wxyz)
        {
            return math.normalize(wxyz).yzwx;
        }

        // Returns three smallest quaternion components in xyz (normalized to 0..1 range), and index/3 of the largest one in w
        public static float4 PackSmallest3Rotation(float4 q)
        {
            // find biggest component
            float4 absQ = math.abs(q);
            int index = 0;
            float maxV = absQ.x;

            if (absQ.y > maxV)
            {
                index = 1;
                maxV = absQ.y;
            }

            if (absQ.z > maxV)
            {
                index = 2;
                maxV = absQ.z;
            }

            if (absQ.w > maxV)
            {
                index = 3;
                maxV = absQ.w;
            }

            q = index switch
            {
                0 => q.yzwx,
                1 => q.xzwy,
                2 => q.xywz,
                _ => q
            };

            float3 three = q.xyz * (q.w >= 0 ? 1 : -1); // -1/sqrt2..+1/sqrt2 range
            three = (three * math.SQRT2) * 0.5f + 0.5f; // 0..1 range

            return new float4(three, index / 3.0f);
        }


        // Based on https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        // Insert two 0 bits after each of the 21 low bits of x
        private static ulong MortonPart1By2(ulong x)
        {
            x &= 0x1fffff;
            x = (x ^ (x << 32)) & 0x1f00000000ffffUL;
            x = (x ^ (x << 16)) & 0x1f0000ff0000ffUL;
            x = (x ^ (x << 8)) & 0x100f00f00f00f00fUL;
            x = (x ^ (x << 4)) & 0x10c30c30c30c30c3UL;
            x = (x ^ (x << 2)) & 0x1249249249249249UL;
            return x;
        }
        // Encode three 21-bit integers into 3D Morton order
        public static ulong MortonEncode3(uint3 v)
        {
            return (MortonPart1By2(v.z) << 2) | (MortonPart1By2(v.y) << 1) | MortonPart1By2(v.x);
        }

        // See GaussianSplatting.hlsl
        public static uint2 DecodeMorton2D_16x16(uint t)
        {
            t = (t & 0xFF) | ((t & 0xFE) << 7); // -EAFBGCHEAFBGCHD
            t &= 0x5555;                        // -E-F-G-H-A-B-C-D
            t = (t ^ (t >> 1)) & 0x3333;        // --EF--GH--AB--CD
            t = (t ^ (t >> 2)) & 0x0f0f;        // ----EFGH----ABCD
            return new uint2(t & 0xF, t >> 8);  // --------EFGHABCD
        }

        public static JsonSerializer GetSplatSerializer()
        {
            var serializer = new JsonSerializer();
            serializer.Formatting = Formatting.Indented;
            return serializer;
        }
    }

    public class JsonVector3Converter : JsonConverter<Vector3>
    {
        public override void WriteJson(JsonWriter writer, Vector3 value, JsonSerializer serializer)
        {
            JObject j = new JObject {{"x", value.x}, {"y", value.y}, {"z", value.z}};

            j.WriteTo(writer);
        }

        //CanRead is false which means the default implementation will be used instead.
        public override Vector3 ReadJson(JsonReader reader, Type objectType, Vector3 existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            return existingValue;
        }

        public override bool CanWrite => true;

        public override bool CanRead => false;
    }

    public class JsonHash128Converter : JsonConverter<Hash128>
    {
        public override void WriteJson(JsonWriter writer, Hash128 value, JsonSerializer serializer)
        {
            JObject j = new JObject {{"hash", value.ToString()}};
            j.WriteTo(writer);
        }

        public override Hash128 ReadJson(JsonReader reader, Type objectType, Hash128 existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            JObject jo = JObject.Load(reader);
            var value = jo.GetValue("hash").ToString();
            return Hash128.Parse(value);
        }
    }
}
