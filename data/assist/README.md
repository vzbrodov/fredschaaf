# Ephemerides for ASSIST

The orbit scripts expect two official JPL files in this directory:

```bash
curl -L --fail --continue-at - \
  https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440 \
  -o data/assist/linux_p1550p2650.440
curl -L --fail --continue-at - \
  https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp \
  -o data/assist/sb441-n16.bsp
(cd data/assist && sha256sum -c SHA256SUMS)
```

The kernels total about 714 MB and are intentionally ignored by Git.
