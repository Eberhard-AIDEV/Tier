#!/bin/bash
# а1
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VER=$(sed -n "s/^VERSION = '\(.*\)'/\1/p" "$HERE/tierlm.py")

stage(){ local root pre app
    root="$1"; pre="${2:-/usr}"; app="$root$pre/lib/tierlm"   # б1
    install -d "$app" "$root$pre/bin" "$root$pre/share/applications" "$root$pre/share/icons/hicolor/scalable/apps"
    install -m644 "$HERE"/{tierlm.py,ui.py,ui.qml,en.txt,glossary.tsv,hub_known.json,tier.cpp,gpu_compat.h,tier_wmma.h,quant.h,quant_tables.h,quant.py,check_quant.py,prepare.py,replay.py,replay.jsonl,chat.py,arch.py,tmpl.py,hub.py,hwprobe.py,tts.py,voice.py,voice_xtts.py,logo.svg,README.md} "$app/"
    install -m644 "$HERE"/example_axioms.jsonl "$HERE"/example_axioms_q.txt "$app/"
    install -m755 "$HERE"/{auto.sh,train.sh,build.sh,check.sh,fetch_model.sh,pydeps.sh} "$app/"
    cp -r "$HERE/stub" "$HERE/stub_cuda" "$HERE/fonts" "$app/"
    install -m644 "$HERE/logo.svg" "$root$pre/share/icons/hicolor/scalable/apps/tierlm.svg"
    # б2
    cat > "$root$pre/bin/tierlm" <<EOF
#!/bin/sh
for p in /usr/bin/python3 python3 python3.14 python3.13 python3.12 python3.11; do
  command -v "\$p" >/dev/null 2>&1 && "\$p" -c "import PySide6.QtQuick, PySide6.QtSvg" 2>/dev/null && exec "\$p" "$pre/lib/tierlm/tierlm.py" "\$@"
done
exec python3 "$pre/lib/tierlm/tierlm.py" "\$@"
EOF
    chmod 755 "$root$pre/bin/tierlm"
    cat > "$root$pre/share/applications/tierlm.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=TierLM
GenericName=Обучение языковых моделей
Comment=Дообучение моделей, которые не помещаются в видеопамять
Exec=tierlm
Icon=tierlm
Terminal=false
Categories=Development;Science;
StartupWMClass=TierLM
EOF
}

deb(){ local d; d=$(mktemp -d)
    stage "$d" /usr
    install -d "$d/DEBIAN"
    cat > "$d/DEBIAN/control" <<EOF
Package: tierlm
Version: $VER
Section: science
Priority: optional
Architecture: all
Depends: python3, gawk, bash, python3-pyside6.qtquick, python3-pyside6.qtqml, python3-pyside6.qtsvg, python3-pyside6.qtwidgets,
 qml6-module-qtquick, qml6-module-qtquick-layouts, qml6-module-qtquick-window
Recommends: libnotify-bin, python3-pip
Maintainer: Eberhard Gottman <so_bad_catty@t.me>
Description: Дообучение моделей, которые не помещаются в видеопамять
 Замороженные веса хранятся в ОЗУ и подаются в видеокарту послойно с опережающей
 подкачкой, поэтому модель 12B в BF16 обучается на 8 ГБ видеопамяти. Изменяются
 собственные веса модели (срез строк каждой матрицы либо целые матрицы выбранных
 слоёв), а не только адаптер; классическая LoRA доступна для сравнения. План
 обучения рассчитывается до запуска. Замороженная часть хранится в BF16 или в
 квантованном виде: собственное квантование либо готовые GGUF и bitsandbytes.
 Движок собирается под видеокарту AMD (ROCm) или nvidia (CUDA) при первом запуске.
EOF
    dpkg-deb --root-owner-group --build "$d" "$HERE/tierlm_${VER}_all.deb" >/dev/null
    rm -rf "$d"; echo "$HERE/tierlm_${VER}_all.deb"
}

case "${1:-}" in
  stage) stage "${2:-}" "${3:-/usr}" ;;
  deb)   deb ;;
  ver)   echo "$VER" ;;
  *) echo "Использование: pack.sh stage <корень> <приставка> | deb | ver"; exit 1 ;;
esac
