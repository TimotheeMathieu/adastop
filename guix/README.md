## Packaging to appimage using guix
Can use

```bash
file=$(guix pack -L . -f appimage --entry-point=bin/adastop python-adastop)
$file --help
```
