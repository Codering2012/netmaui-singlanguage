using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace SignLanguageApi.Services
{
    public static class EncryptionHelper
    {
        private static readonly byte[] Key = Encoding.ASCII.GetBytes("SignLangKey_2026_Safe_32_Bytes_!"); // 32 bytes
        private static readonly byte[] Iv = Encoding.ASCII.GetBytes("SignLang_Iv_16ch"); // 16 bytes

        public static string Encrypt(string plainText)
        {
            if (string.IsNullOrEmpty(plainText)) return plainText;

            using var aes = Aes.Create();
            aes.Key = Key;
            aes.IV = Iv;

            var encryptor = aes.CreateEncryptor(aes.Key, aes.IV);
            using var ms = new MemoryStream();
            using (var cs = new CryptoStream(ms, encryptor, CryptoStreamMode.Write))
            {
                using (var sw = new StreamWriter(cs))
                {
                    sw.Write(plainText);
                }
            }
            return Convert.ToBase64String(ms.ToArray());
        }

        public static string Decrypt(string cipherText)
        {
            if (string.IsNullOrEmpty(cipherText)) return cipherText;

            try
            {
                using var aes = Aes.Create();
                aes.Key = Key;
                aes.IV = Iv;

                var decryptor = aes.CreateDecryptor(aes.Key, aes.IV);
                using var ms = new MemoryStream(Convert.FromBase64String(cipherText));
                using var cs = new CryptoStream(ms, decryptor, CryptoStreamMode.Read);
                using var sr = new StreamReader(cs);
                return sr.ReadToEnd();
            }
            catch
            {
                return cipherText; // Return original if decryption fails (e.g. not encrypted yet)
            }
        }
    }
}
