"""Acceso a las trayectorias crudas de HTP-MD sin descargarlas enteras: fichero remoto con peticiones Range sobre la URL firmada de S3.

Las URL firmadas caducan (~1 h) y la API a veces devuelve una ya caducada: Remoto pide otra automáticamente cuando S3 responde 403,
también a mitad de una lectura larga."""
import io, json, time, urllib.error, urllib.request

def url(i):
    """URL firmada (caduca) de la trayectoria cruda de la muestra i."""
    # La API va detrás de una caché (CloudFront) que guarda la respuesta más tiempo del que dura la URL firmada (1 h):
    # el parámetro con la hora fuerza una respuesta nueva.
    r = urllib.request.Request(f"https://www.htpmd.matr.io/api/trajectories/{i}/raw?_={time.time_ns()}", headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(r, timeout=40))["s3_presigned_url"]

class Remoto(io.RawIOBase):
    def __init__(s, muestra):
        s.muestra, s.url, s.pos = muestra, None, 0
        s.size = int(s._pedir(0, 0).headers["Content-Range"].split("/")[1])

    def _pedir(s, a, b):
        """GET del rango [a, b]; renueva la URL si falta o ha caducado y reintenta ante fallos de red."""
        ultimo = None
        for intento in range(10):
            try:
                if s.url is None: s.url = url(s.muestra)
                return urllib.request.urlopen(urllib.request.Request(s.url, headers={"Range": f"bytes={a}-{b}"}), timeout=180)
            except urllib.error.HTTPError as e:
                ultimo = e
                if e.code == 403: s.url = None; time.sleep(3 + 2 * intento)        # caducada: pedir otra
                else: time.sleep(5 + 5 * intento)
            except Exception as e:
                ultimo = e; time.sleep(5 + 5 * intento)
        raise ultimo

    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, whence=0):
        s.pos = off if whence == 0 else s.pos + off if whence == 1 else s.size + off; return s.pos
    def readinto(s, b):
        n = min(len(b), s.size - s.pos)
        if n <= 0: return 0
        for intento in range(5):
            try:
                d = s._pedir(s.pos, s.pos + n - 1).read(); break
            except Exception:
                if intento == 4: raise
                time.sleep(5)
        b[:len(d)] = d; s.pos += len(d); return len(d)
