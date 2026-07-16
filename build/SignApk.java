// Minimal apksig driver (see BUILD.md "sign v2+v3 with apksig"):
//   java -cp apksig.jar SignApk.java verify <apk>
//   java -cp apksig.jar SignApk.java sign <in.apk> <out.apk> <keystore> <storepass> <alias>
import com.android.apksig.ApkSigner;
import com.android.apksig.ApkVerifier;
import java.io.*;
import java.security.*;
import java.security.cert.X509Certificate;
import java.util.*;

public class SignApk {
  static String fp(X509Certificate c) throws Exception {
    byte[] d = MessageDigest.getInstance("SHA-256").digest(c.getEncoded());
    StringBuilder sb = new StringBuilder();
    for (byte b : d) sb.append(String.format("%02X:", b));
    return sb.substring(0, sb.length() - 1);
  }
  public static void main(String[] a) throws Exception {
    if (a[0].equals("verify")) {
      ApkVerifier.Result r = new ApkVerifier.Builder(new File(a[1])).build().verify();
      System.out.println("verified=" + r.isVerified() + " v2=" + r.isVerifiedUsingV2Scheme() + " v3=" + r.isVerifiedUsingV3Scheme());
      for (X509Certificate c : r.getSignerCertificates()) System.out.println("signer SHA256: " + fp(c));
      for (ApkVerifier.IssueWithParams e : r.getErrors()) System.out.println("ERROR: " + e);
      return;
    }
    KeyStore ks = KeyStore.getInstance("PKCS12");
    try (InputStream is = new FileInputStream(a[3])) { ks.load(is, a[4].toCharArray()); }
    PrivateKey key = (PrivateKey) ks.getKey(a[5], a[4].toCharArray());
    List<X509Certificate> certs = new ArrayList<>();
    for (java.security.cert.Certificate c : ks.getCertificateChain(a[5])) certs.add((X509Certificate) c);
    ApkSigner.SignerConfig sc = new ApkSigner.SignerConfig.Builder("signer", key, certs).build();
    new ApkSigner.Builder(Collections.singletonList(sc))
      .setInputApk(new File(a[1])).setOutputApk(new File(a[2]))
      .setV1SigningEnabled(false).setV2SigningEnabled(true).setV3SigningEnabled(true)
      .build().sign();
    System.out.println("signed -> " + a[2] + "  key: " + fp(certs.get(0)));
  }
}
