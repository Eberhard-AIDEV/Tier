#!/bin/bash
# а1
set -u
SRC="$(cd "$(dirname "$0")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/tierlm"
say(){ printf '\n\033[1;96m%s\033[0m\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }
SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=

# г1
case "${1:-}" in --remove|--purge)
    if have pacman && pacman -Qq tierlm >/dev/null 2>&1; then $SUDO pacman -R --noconfirm tierlm
    elif have dpkg && dpkg -s tierlm >/dev/null 2>&1; then $SUDO apt-get remove -y tierlm
    else rm -rf "$HOME/.local/lib/tierlm" "$HOME/.local/bin/tierlm" \
                "$HOME/.local/share/applications/tierlm.desktop" \
                "$HOME/.local/share/icons/hicolor/scalable/apps/tierlm.svg"; fi
    if [ "$1" = "--purge" ]; then rm -rf "$DATA"; echo "Удалено всё, включая веса и результаты обучения."
    else echo "Программа удалена. Веса и результаты обучения сохранены в $DATA."; fi
    exit 0 ;;
esac

# д1
if [ -f "$DATA/train.pid" ]; then
    P=$(head -n1 "$DATA/train.pid" 2>/dev/null); N=$(sed -n 2p "$DATA/train.pid" 2>/dev/null)
    if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then
        echo "Выполняется обучение «$N»: обновление заменило бы его скрипты во время работы."
        echo "Для установки требуется дождаться окончания обучения или остановить его в окне программы (продолжить можно будет со снимка весов), затем повторить установку."
        exit 1
    fi
fi

# е1
if have pacman && have makepkg; then
    say "Установка пакетом через pacman"
    ( cd "$SRC" && makepkg -sif --noconfirm ) || exit 1
    HOW="pacman -R tierlm"
elif have apt-get && have dpkg-deb; then
    say "Сборка пакета и установка через apt"
    DEB=$(bash "$SRC/pack.sh" deb) || exit 1
    $SUDO apt-get install -y --reinstall "$DEB" || exit 1     # е2
    HOW="apt remove tierlm"
else
    say "Пакетный менеджер не определён, выполняется установка в домашнюю папку"
    bash "$SRC/pack.sh" stage "" "$HOME/.local" || exit 1
    HOW="bash install.sh --remove"
    case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
        echo "Требуется добавить в ~/.bashrc строку:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;; esac
fi
have update-desktop-database && $SUDO update-desktop-database >/dev/null 2>&1
have gtk-update-icon-cache && $SUDO gtk-update-icon-cache -q -f /usr/share/icons/hicolor >/dev/null 2>&1

# ж1
GPU=$(bash "$SRC/build.sh" --vendor)
if [ "$GPU" = amd ] && ! have hipcc && [ ! -x /opt/rocm/bin/hipcc ]; then
    say "Установка ROCm (без него движок под AMD не собирается)"
    # ж2
    if have pacman; then $SUDO pacman -S --needed --noconfirm rocm-hip-sdk hipblas && $SUDO pacman -S --needed --noconfirm hipblaslt
    elif have apt-get; then $SUDO apt-get install -y hipcc libhipblas-dev && { $SUDO apt-get install -y libhipblaslt-dev 2>/dev/null || true; }
    elif have dnf; then $SUDO dnf install -y rocm-hip-devel hipblas-devel && { $SUDO dnf install -y hipblaslt-devel 2>/dev/null || true; }
    elif have zypper; then echo "  openSUSE: ROCm устанавливается из репозитория AMD — https://rocm.docs.amd.com"; fi
elif [ "$GPU" = nvidia ] && ! have nvcc && [ ! -x /usr/local/cuda/bin/nvcc ] && [ ! -x /opt/cuda/bin/nvcc ]; then
    say "Установка CUDA Toolkit (без него движок под nvidia не собирается)"
    if have pacman; then $SUDO pacman -S --needed --noconfirm cuda
    elif have apt-get; then $SUDO apt-get install -y nvidia-cuda-toolkit
    elif have dnf; then $SUDO dnf install -y cuda-toolkit
    elif have zypper; then echo "  openSUSE: CUDA устанавливается из репозитория nvidia — https://developer.nvidia.com/cuda-downloads"; fi
fi

# з1
if ! { /usr/bin/python3 -c 'import PySide6.QtQuick, PySide6.QtSvg' 2>/dev/null || python3 -c 'import PySide6.QtQuick, PySide6.QtSvg' 2>/dev/null; }; then
    say "Для окна программы требуется PySide6 (Qt 6, Qt Quick)"
    if have dnf; then echo "  sudo dnf install python3-pyside6"
    elif have zypper; then echo "  sudo zypper install python3-pyside6"
    else echo "  python3 -m pip install --user PySide6"; fi
    echo "  Без окна: tierlm сервер — панель в браузере"
fi

say "TierLM установлен"
echo "  Запуск:    TierLM в меню программ или команда tierlm"
echo "  Удаление:  $HOW"
echo "  Данные:    $DATA   (веса, результаты обучения, собранный движок)"
echo
echo "Разделы слева: Новое обучение, Ход обучения, Запуски, Разговор, Проверка, Настройки."
echo "Недостающее отображается в проверках у кнопки «Начать обучение»: сборка движка, выбор или загрузка весов."
echo "Распакованную папку после установки можно удалить."
