// а1
import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import Tier 1.0

Window {
    id: win
    width: 1380; height: 880
    minimumWidth: 760; minimumHeight: 520
    visible: true
    title: tv && tv.live && tv.doc ? "TierLM · " + tv.doc : "TierLM"
    color: "transparent"
    flags: Qt.Window | Qt.FramelessWindowHint

    // а2
    property var hh: B.health
    property var sv: B.sv
    property var tv: B.tv
    property var wv: B.wv
    property var cm: B.cmp
    property var jb: B.job
    property var bn: B.bench
    property int rev: B.rev
    readonly property bool en: B.lang() === "en"
    property string view: "setup"
    property string sec: "look"
    readonly property bool flat: visibility === Window.Maximized || visibility === Window.FullScreen
    readonly property bool compact: width < 960
    readonly property bool typing: activeFocusItem !== null && activeFocusItem.hasOwnProperty("cursorPosition")
    property var cfg: B.settings()
    property var ap: mergeAP(cfg["свои"]["APPEAR"])
    property var ui: mergeUI(cfg["свои"]["UI"])
    property string bgPath: cfg["свои"]["BG"] || ""
    property string liveName: hh && hh["идёт"] ? hh["идёт"] : ""
    property var liveRow: { if (!hh || !liveName) return null; const L = hh["запуски"] || []; for (let i = 0; i < L.length; i++) if (L[i]["имя"] === liveName) return L[i]; return null }

    function tx(s) { return en ? B.loc(s) : s }
    function famOf(k) { return ({inter: "TL Inter", golos: "TL Golos", onest: "TL Onest", manrope: "TL Manrope", jost: "TL Jost", geologica: "TL Geologica",
                                 system: Qt.application.font.family, mono: "monospace"})[k] || "TL Inter" }
    readonly property var apDef: ({theme: "dark", bg: "aurora", solid: "#0e0f13", dim: .35, iblur: 0, accent: "#0a84ff", radius: 16, line: .5, glass: .62,
                                   ff: "inter", fs: 13.5, weight: 400, density: 1, side: "left", sidew: 248, wide: 0, full: "gap", motion: 1, lw: 1.75, cfill: 1, cgrid: 1, cdots: 1})
    readonly property var uiDef: ({start: "auto", notify: 1, tray: 1, confirmStop: 1, dec: "dot", charts: ["loss", "acc", "vram", "temp", "time", "grad"], ccols: 0, ch: 150,
                                   chatTemp: .5, chatMax: 200, chatScene: "", chatScale: 1, chatTopK: 0, chatTopP: 1, chatMinP: 0, chatRep: 1, chatSeed: -1, chatCtx: B.ctxDefault(), chatPres: 0, chatFreq: 0, chatRepN: 256, chatThink: false, voiceProv: "", voiceMode: "stream", voiceXtts: "", voiceSpk: "", voiceLang: "ru", voiceDev: "auto", presets: []})
    function parse(j) { try { return j ? JSON.parse(j) : {} } catch (e) { return {} } }
    function mergeAP(j) { const a = Object.assign({}, apDef, parse(j)); if (!famOf(a.ff)) a.ff = "inter"; if (a.weight > 500) a.weight = 500; if (a.weight < 400) a.weight = 400; return a }
    function mergeUI(j) { return Object.assign({}, uiDef, parse(j)) }
    function apSet(p) { ap = Object.assign({}, ap, p); apSave.restart() }
    function uiSet(p) { ui = Object.assign({}, ui, p); uiSave.restart() }
    // л1
    readonly property string curModel: hh && hh["модель"] && hh["модель"]["есть"] ? (hh["модель"]["папка"] || "") : ""
    onCurModelChanged: if (curModel && ui.chatModel !== curModel) Qt.callLater(() => uiSet(Object.assign(chatView.authorsPreset(), {chatModel: curModel, chatCtx: Math.min(+ui.chatCtx || B.ctxDefault(), B.ctxMax())})))
    Timer { id: apSave; interval: 300; onTriggered: { B.saveAppear(JSON.stringify(ap)); B.saveSettings({THEME: ap.theme === "light" ? "light" : ""}) } }
    Timer { id: uiSave; interval: 300; onTriggered: B.saveUI(JSON.stringify(ui)) }
    readonly property var presets: [
        [tx("Ночь"),     {theme: "dark", bg: "aurora", accent: "#0a84ff", radius: 16, line: .5, glass: .62, ff: "inter", fs: 13.5, weight: 400, density: 1}],
        [tx("Океан"),    {theme: "dark", bg: "ocean", accent: "#64d2ff", radius: 18, line: .5, glass: .55, ff: "onest", fs: 13.5, weight: 400, density: 1}],
        [tx("Графит"),   {theme: "dark", bg: "ink", accent: "#e5e5ea", radius: 12, line: 1, glass: .88, ff: "inter", fs: 13.5, weight: 400, density: 1}],
        [tx("Угли"),     {theme: "dark", bg: "ember", accent: "#ff9f0a", radius: 16, line: .5, glass: .6, ff: "manrope", fs: 13.5, weight: 400, density: 1}],
        [tx("Терминал"), {theme: "dark", bg: "plain", accent: "#30d158", radius: 6, line: 1, glass: 1, ff: "mono", fs: 13, weight: 400, density: .85}],
        [tx("Контраст"), {theme: "dark", bg: "plain", accent: "#ffd60a", radius: 10, line: 1.5, glass: 1, ff: "inter", fs: 14.5, weight: 500, density: 1}],
        [tx("Пудра"),    {theme: "light", bg: "dawn", accent: "#c06c84", radius: 20, line: .5, glass: .72, ff: "golos", fs: 13.5, weight: 400, density: 1.15}],
        [tx("Бумага"),   {theme: "light", bg: "solid", solid: "#f5f2ec", accent: "#1d1d1f", radius: 10, line: .5, glass: .92, ff: "geologica", fs: 14, weight: 400, density: 1}],
        [tx("Туман"),    {theme: "light", bg: "mist", accent: "#007aff", radius: 16, line: .5, glass: .6, ff: "inter", fs: 13.5, weight: 400, density: 1}],
        [tx("Лаванда"),  {theme: "light", bg: "lavender", accent: "#5e5ce6", radius: 18, line: .5, glass: .66, ff: "onest", fs: 13.5, weight: 400, density: 1}]]
    readonly property var bgs: [["aurora", tx("Сияние"), "#1d2b5a", "#d8e3ff"], ["ocean", tx("Океан"), "#0b3d5c", "#cfe9f7"], ["dawn", tx("Рассвет"), "#4a2a3a", "#fbe1e6"],
                                ["ember", tx("Угли"), "#4d1d0e", "#fde3d3"], ["lavender", tx("Лаванда"), "#30295c", "#e6e1fb"], ["mist", tx("Туман"), "#1a1d22", "#e3e8ee"],
                                ["ink", tx("Тушь"), "#24262b", "#ffffff"], ["plain", tx("Ровный"), "#0b0c10", "#f5f5f7"]]

    // э1
    QtObject {
        id: th
        readonly property bool light: ap.theme === "light"
        readonly property color accent: ap.accent || "#0a84ff"
        readonly property color ink: light ? "#1d1d1f" : Qt.rgba(1, 1, 1, .93)
        readonly property color ink2: light ? Qt.rgba(.235, .235, .263, .76) : Qt.rgba(.922, .922, .961, .64)
        readonly property color ink3: light ? Qt.rgba(.235, .235, .263, .54) : Qt.rgba(.922, .922, .961, .42)
        readonly property color ink4: light ? Qt.rgba(.235, .235, .263, .30) : Qt.rgba(.922, .922, .961, .22)
        readonly property color sep: light ? Qt.rgba(0, 0, 0, .075) : Qt.rgba(1, 1, 1, .075)
        readonly property color edge: light ? Qt.rgba(0, 0, 0, .08) : Qt.rgba(1, 1, 1, .085)
        readonly property color edge2: light ? Qt.rgba(0, 0, 0, .11) : Qt.rgba(1, 1, 1, .12)
        readonly property color fill: light ? Qt.rgba(0, 0, 0, .045) : Qt.rgba(1, 1, 1, .065)
        readonly property color fill2: light ? Qt.rgba(0, 0, 0, .075) : Qt.rgba(1, 1, 1, .10)
        readonly property color fill3: light ? Qt.rgba(0, 0, 0, .11) : Qt.rgba(1, 1, 1, .15)
        readonly property color field: light ? Qt.rgba(1, 1, 1, .8) : Qt.rgba(0, 0, 0, .2)
        readonly property color segOn: light ? "#ffffff" : Qt.rgba(1, 1, 1, .17)
        readonly property color solid: light ? "#ffffff" : "#2b2b31"
        readonly property color scrim: light ? Qt.rgba(.08, .08, .1, .24) : Qt.rgba(0, 0, 0, .46)
        readonly property color ok: light ? "#1f9d3a" : "#32d74b"
        readonly property color warn: light ? "#b97400" : "#ffd60a"
        readonly property color bad: light ? "#e0352b" : "#ff453a"
        readonly property color okW: Qt.rgba(ok.r, ok.g, ok.b, light ? .12 : .15)
        readonly property color warnW: Qt.rgba(warn.r, warn.g, warn.b, light ? .12 : .15)
        readonly property color badW: Qt.rgba(bad.r, bad.g, bad.b, light ? .10 : .15)
        readonly property color wash: Qt.rgba(accent.r, accent.g, accent.b, light ? .10 : .16)
        readonly property color wash2: Qt.rgba(accent.r, accent.g, accent.b, light ? .17 : .26)
        readonly property color accentInk: light ? Qt.rgba(accent.r * .8, accent.g * .8, accent.b * .8, 1) : Qt.rgba(accent.r * .78 + .22, accent.g * .78 + .22, accent.b * .78 + .22, 1)
        readonly property real lumA: .2126 * accent.r + .7152 * accent.g + .0722 * accent.b
        readonly property color onAccent: lumA > .62 ? "#111114" : "#ffffff"
        readonly property real al: Math.min(1, .3 + .68 * ap.glass)
        readonly property color card: light ? Qt.rgba(1, 1, 1, al) : Qt.rgba(40 / 255, 40 / 255, 46 / 255, al)
        readonly property color mat: light ? Qt.rgba(246 / 255, 246 / 255, 249 / 255, Math.min(1, al * .92)) : Qt.rgba(26 / 255, 26 / 255, 31 / 255, Math.min(1, al * .95))
        readonly property color pop: light ? Qt.rgba(252 / 255, 252 / 255, 253 / 255, .985) : Qt.rgba(44 / 255, 44 / 255, 50 / 255, .985)
        readonly property real r: ap.radius
        readonly property real r2: Math.round(r * .62)
        readonly property real rc: Math.max(4, Math.min(12, Math.round(r * .5)))
        readonly property real hair: ap.line
        readonly property string ff: famOf(ap.ff)
        readonly property real fs: ap.fs
        readonly property int wt: ap.weight
        readonly property int wth: Math.min(wt + 200, 700)
        readonly property real d: ap.density
        readonly property int pad: Math.round(12 * d)
        readonly property int gap: Math.round(12 * d)
        readonly property bool motion: !!ap.motion
        readonly property color s2: light ? "#d67a00" : "#ff9f0a"
        readonly property color s3: light ? "#d63864" : "#ff6482"
        readonly property color s4: light ? "#0a8fc8" : "#64d2ff"
        readonly property color s5: light ? "#9a3cd0" : "#bf5af2"
        readonly property color s6: light ? "#1f9d3a" : "#30d158"
        function sc(c) { return c === "s2" ? s2 : c === "s3" ? s3 : c === "s4" ? s4 : c === "s5" ? s5 : c === "s6" ? s6 : accent }
    }

    // ю1
    component Txt: Text {
        property bool gloss: true   // ю2
        color: th.ink; font.family: th.ff; font.pixelSize: th.fs; font.weight: th.wt
        renderType: Text.NativeRendering; textFormat: Text.PlainText; elide: Text.ElideRight; verticalAlignment: Text.AlignVCenter
        HoverHandler {
            id: glh
            enabled: parent.gloss && parent.text.length > 1 && parent.text.length < 160 && parent.visible
            onHoveredChanged: { if (hovered) { const g = B.gloss(parent.text); if (g !== "") tipLayer.armLow(parent, g) } else tipLayer.disarmLow(parent) }
        }
    }
    component Rich: Txt { textFormat: Text.StyledText; elide: Text.ElideNone }
    component H2: Txt { font.pixelSize: th.fs + 3; font.weight: th.wth }
    component Small: Txt { font.pixelSize: th.fs - 2; color: th.ink3 }
    component Mono: TextEdit {   // ав1
        color: th.ink2; font.family: "monospace"; font.pixelSize: th.fs - 1.5; renderType: TextEdit.NativeRendering; wrapMode: TextEdit.WrapAnywhere; textFormat: TextEdit.PlainText
        readOnly: true; selectByMouse: true; selectionColor: th.wash2; selectedTextColor: th.ink; persistentSelection: true
    }

    component Tip: HoverHandler {
        id: tp
        property string text: ""
        onHoveredChanged: hovered && text !== "" ? tipLayer.arm(tp.parent, text) : tipLayer.disarm(tp.parent)
        onTextChanged: if (hovered && tipLayer.tgt === tp.parent) tipLayer.txt = text
    }

    component Btn: Rectangle {
        id: bt
        property string text: ""
        property string icon: ""
        property string kind: ""            // ад1
        property string size: ""            // ад2
        property bool on: true
        property string tip: ""
        property string kbd: ""
        readonly property bool iconOnly: text === ""
        readonly property bool hov: ma.containsMouse && on
        readonly property bool down: ma.pressed
        signal clicked()
        implicitHeight: size === "sm" ? Math.round(24 + 2 * th.d) : size === "lg" ? Math.round(38 + 4 * th.d) : Math.round(28 + 4 * th.d)
        implicitWidth: iconOnly ? implicitHeight : lay.implicitWidth + (size === "sm" ? 18 : size === "lg" ? 34 : 24)
        radius: th.rc
        opacity: on ? 1 : .42
        color: kind === "primary" ? (down ? Qt.darker(th.accent, 1.12) : hov ? Qt.lighter(th.accent, 1.07) : th.accent)
             : kind === "tint" ? (down ? th.wash2 : hov ? th.wash2 : th.wash)
             : kind === "quiet" ? (down ? th.fill2 : hov ? th.fill : "transparent")
             : kind === "danger" ? (down ? th.badW : hov ? th.badW : "transparent")
             : kind === "solidDanger" ? (down ? Qt.darker(th.bad, 1.1) : hov ? Qt.lighter(th.bad, 1.06) : th.bad)
             : (down ? th.fill3 : hov ? th.fill2 : th.fill)
        readonly property color fg: kind === "primary" ? th.onAccent : kind === "tint" ? th.accentInk : kind === "danger" ? th.bad
                                  : kind === "solidDanger" ? "#ffffff" : kind === "quiet" ? (hov ? th.ink : th.ink2) : th.ink
        Behavior on color { enabled: th.motion; ColorAnimation { duration: 90 } }
        RowLayout {
            id: lay; anchors.centerIn: parent; spacing: 6
            Icon { visible: bt.icon !== ""; name: bt.icon; color: bt.fg; Layout.preferredWidth: bt.size === "lg" ? 16 : 15; Layout.preferredHeight: Layout.preferredWidth; sw: 1.6 }
            Txt {
                visible: bt.text !== ""; text: bt.text; color: bt.fg
                font.pixelSize: bt.size === "sm" ? th.fs - 1 : bt.size === "lg" ? th.fs + 1 : th.fs - .5
                font.weight: bt.kind === "primary" || bt.size === "lg" ? th.wth : Math.min(th.wt + 100, 600)
            }
            Rectangle {
                visible: bt.kbd !== ""; radius: 4; color: Qt.rgba(bt.fg.r, bt.fg.g, bt.fg.b, .18)
                Layout.preferredHeight: 18; Layout.preferredWidth: kt.implicitWidth + 10; Layout.leftMargin: 4
                Txt { id: kt; anchors.centerIn: parent; text: bt.kbd; color: bt.fg; font.pixelSize: th.fs - 3; opacity: .85 }
            }
        }
        MouseArea { id: ma; anchors.fill: parent; hoverEnabled: true; enabled: bt.on; cursorShape: Qt.PointingHandCursor; onClicked: { tipLayer.hideNow(); bt.clicked() } }
        Tip { text: bt.tip }
    }

    component Pill: Rectangle {
        id: pl
        property string kind: ""
        property string text: ""
        property string tip: ""
        readonly property color fg: kind === "ok" ? th.ok : kind === "bad" ? th.bad : (kind === "warn" || kind === "need") ? th.warn
                                  : (kind === "live" || kind === "wait") ? th.accentInk : th.ink2
        visible: text !== ""
        implicitHeight: Math.round(20 + 2 * th.d); implicitWidth: prow.implicitWidth + 16
        radius: height / 2
        color: kind === "ok" ? th.okW : kind === "bad" ? th.badW : (kind === "warn" || kind === "need") ? th.warnW : (kind === "live" || kind === "wait") ? th.wash : th.fill
        Row {
            id: prow; anchors.centerIn: parent; spacing: 6
            Rectangle {
                width: 6; height: 6; radius: 3; color: pl.fg; anchors.verticalCenter: parent.verticalCenter
                SequentialAnimation on opacity {
                    running: th.motion && (pl.kind === "live" || pl.kind === "wait"); loops: Animation.Infinite
                    NumberAnimation { to: .3; duration: 1000; easing.type: Easing.InOutSine }
                    NumberAnimation { to: 1; duration: 1000; easing.type: Easing.InOutSine }
                }
            }
            Txt { text: pl.text; color: pl.fg; font.pixelSize: th.fs - 2; font.weight: Math.min(th.wt + 100, 600) }
        }
        Tip { text: pl.tip }
    }

    component Chip: Rectangle {
        id: cp
        property string text: ""
        property string tip: ""
        property color tint: th.fill
        property color fg: th.ink2
        implicitHeight: Math.round(21 + 2 * th.d); implicitWidth: ct.implicitWidth + 16
        radius: height / 2; color: tint
        Rich { id: ct; anchors.centerIn: parent; text: cp.text; color: cp.fg; font.pixelSize: th.fs - 2 }
        Tip { text: cp.tip }
    }

    component Seg: Rectangle {
        id: sg
        property var options: []
        property var value
        property bool small: false
        signal picked(var v)
        implicitHeight: small ? 26 : Math.round(28 + 3 * th.d)
        implicitWidth: srow.implicitWidth + 6
        radius: th.rc; color: th.fill
        Row {
            id: srow; x: 3; y: 3; height: sg.height - 6; spacing: 2
            Repeater {
                model: sg.options
                delegate: Rectangle {
                    required property var modelData
                    readonly property bool sel: String(modelData[0]) === String(sg.value)
                    height: srow.height; width: stx.implicitWidth + (sg.small ? 16 : 22)
                    radius: Math.max(3, th.rc - 2)
                    color: sel ? th.segOn : sm.containsMouse ? th.fill : "transparent"
                    border.width: sel && th.light ? 1 : 0; border.color: th.edge
                    Txt {
                        id: stx; anchors.centerIn: parent; text: parent.modelData[1]
                        color: parent.sel ? th.ink : th.ink2; font.pixelSize: sg.small ? th.fs - 2 : th.fs - 1
                        font.weight: parent.sel ? Math.min(th.wt + 150, 650) : th.wt
                    }
                    MouseArea { id: sm; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: sg.picked(parent.modelData[0]) }
                }
            }
        }
    }

    component Sw: Rectangle {
        id: swc
        property bool checked: false
        property bool on: true
        signal toggled(bool v)
        implicitWidth: 36; implicitHeight: 21; radius: height / 2
        color: checked ? th.accent : th.fill3
        opacity: on ? 1 : .42
        Behavior on color { enabled: th.motion; ColorAnimation { duration: 120 } }
        Rectangle {
            width: 17; height: 17; radius: 8.5; y: 2; x: swc.checked ? swc.width - width - 2 : 2; color: "#ffffff"
            border.width: th.light ? 1 : 0; border.color: Qt.rgba(0, 0, 0, .08)
            Behavior on x { enabled: th.motion; NumberAnimation { duration: 140; easing.type: Easing.OutCubic } }
        }
        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; enabled: swc.on; onClicked: swc.toggled(!swc.checked) }
    }

    component Sld: Item {
        id: sl
        property real from: 0
        property real to: 1
        property real step: .01
        property real value: 0
        property string out: ""
        signal moved(real v)
        implicitHeight: 26; implicitWidth: 260
        Item {
            id: trk; anchors.left: parent.left; anchors.right: outT.left; anchors.rightMargin: 12; height: parent.height
            Rectangle { anchors.verticalCenter: parent.verticalCenter; width: parent.width; height: 4; radius: 2; color: th.fill3 }
            Rectangle { anchors.verticalCenter: parent.verticalCenter; width: knob.x + knob.width / 2; height: 4; radius: 2; color: th.accent }
            Rectangle {
                id: knob; width: 17; height: 17; radius: 8.5; color: "#ffffff"; anchors.verticalCenter: parent.verticalCenter
                border.width: 1; border.color: Qt.rgba(0, 0, 0, th.light ? .14 : .3)
                x: Math.max(0, Math.min(1, (sl.value - sl.from) / ((sl.to - sl.from) || 1))) * (trk.width - width)
            }
            MouseArea {
                anchors.fill: parent; cursorShape: Qt.PointingHandCursor; preventStealing: true
                function setX(mx) {
                    const f = Math.max(0, Math.min(1, (mx - knob.width / 2) / ((trk.width - knob.width) || 1)))
                    let v = sl.from + f * (sl.to - sl.from)
                    v = Math.round(v / sl.step) * sl.step
                    v = +v.toFixed(4)
                    if (v !== sl.value) sl.moved(v)
                }
                onPressed: (e) => setX(e.x)
                onPositionChanged: (e) => { if (pressed) setX(e.x) }
            }
        }
        Txt { id: outT; anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; width: 58; horizontalAlignment: Text.AlignRight; text: sl.out; color: th.ink3; font.pixelSize: th.fs - 1.5 }
    }

    component Field: Rectangle {
        id: fd
        property alias text: ti.text
        property alias input: ti
        property string ph: ""
        property bool mono: false
        property bool bad: false
        property bool ro: false
        property string icon: ""
        property int halign: TextInput.AlignLeft
        signal edited(string t)
        signal accepted()
        signal done(string t)
        implicitHeight: Math.round(28 + 4 * th.d); implicitWidth: 220
        radius: th.rc; color: th.field
        border.width: ti.activeFocus ? 1.5 : 1
        border.color: fd.bad ? th.bad : ti.activeFocus ? th.accent : th.edge2
        Icon { id: fic; visible: fd.icon !== ""; name: fd.icon; color: th.ink3; width: 15; height: 15; x: 10; anchors.verticalCenter: parent.verticalCenter }
        TextInput {
            id: ti
            anchors.fill: parent; anchors.leftMargin: fd.icon !== "" ? 32 : 10; anchors.rightMargin: 10
            verticalAlignment: TextInput.AlignVCenter; horizontalAlignment: fd.halign
            color: th.ink; selectionColor: th.wash2; selectedTextColor: th.ink
            font.family: fd.mono ? "monospace" : th.ff; font.pixelSize: fd.mono ? th.fs - 1 : th.fs
            clip: true; selectByMouse: true; readOnly: fd.ro; renderType: TextInput.NativeRendering
            onTextEdited: fd.edited(text)
            onAccepted: { fd.accepted(); fd.done(text) }
            onEditingFinished: fd.done(text)
        }
        Txt {
            anchors.fill: ti; text: fd.ph; color: th.ink4; visible: ti.text === "" && !ti.inputMethodComposing
            horizontalAlignment: fd.halign === TextInput.AlignRight ? Text.AlignRight : Text.AlignLeft
        }
        MouseArea { anchors.fill: parent; cursorShape: Qt.IBeamCursor; acceptedButtons: Qt.NoButton }
    }

    component Area: Rectangle {
        id: ar
        property alias text: te.text
        property alias edit: te
        property string ph: ""
        property bool mono: false
        property bool grow: false
        property real minH: 96
        property real maxH: 240
        property bool enterSends: false        // ам1
        signal edited()
        signal submit()
        radius: th.rc; color: th.field
        border.width: te.activeFocus ? 1.5 : 1; border.color: te.activeFocus ? th.accent : th.edge2
        implicitHeight: grow ? Math.min(maxH, Math.max(minH, te.contentHeight + 18)) : minH
        Flickable {
            id: af; anchors.fill: parent; anchors.margins: 9; anchors.rightMargin: 6
            contentWidth: width; contentHeight: te.contentHeight; clip: true; interactive: false; boundsBehavior: Flickable.StopAtBounds
            TextEdit {
                id: te; width: af.width - 4; wrapMode: TextEdit.Wrap
                color: th.ink; selectionColor: th.wash2; selectedTextColor: th.ink
                font.family: ar.mono ? "monospace" : th.ff; font.pixelSize: ar.mono ? th.fs - 1 : th.fs
                selectByMouse: true; renderType: TextEdit.NativeRendering
                onTextChanged: ar.edited()
                Keys.onPressed: (e) => {
                    if (ar.enterSends && (e.key === Qt.Key_Return || e.key === Qt.Key_Enter) && !(e.modifiers & (Qt.ShiftModifier | Qt.ControlModifier))) { e.accepted = true; ar.submit() }
                }
                onCursorRectangleChanged: {
                    if (cursorRectangle.y < af.contentY) af.contentY = cursorRectangle.y
                    else if (cursorRectangle.y + cursorRectangle.height > af.contentY + af.height) af.contentY = cursorRectangle.y + cursorRectangle.height - af.height
                }
            }
        }
        WheelHandler {
            onWheel: (ev) => { const d = ev.pixelDelta.y !== 0 ? ev.pixelDelta.y : ev.angleDelta.y / 120 * 48
                               af.contentY = Math.max(0, Math.min(Math.max(0, af.contentHeight - af.height), af.contentY - d)) }
        }
        Txt { x: 10; y: 9; width: parent.width - 20; text: ar.ph; color: th.ink4; visible: te.text === ""; wrapMode: Text.Wrap; elide: Text.ElideNone; verticalAlignment: Text.AlignTop }
        MouseArea { anchors.fill: parent; cursorShape: Qt.IBeamCursor; acceptedButtons: Qt.NoButton }
    }

    component Combo: Rectangle {
        id: cb
        property var options: []
        property var value
        property string ph: ""
        signal picked(var v)
        readonly property string label: { for (let i = 0; i < options.length; i++) if (String(options[i][0]) === String(value)) return options[i][1]; return value === undefined || value === null ? "" : String(value) }
        implicitHeight: Math.round(28 + 4 * th.d); implicitWidth: 220
        radius: th.rc; color: cma.containsMouse ? th.fill2 : th.field
        border.width: 1; border.color: menu.owner === cb ? th.accent : th.edge2
        Txt { anchors.left: parent.left; anchors.leftMargin: 10; anchors.right: chev.left; anchors.rightMargin: 6; anchors.verticalCenter: parent.verticalCenter
              text: cb.label || cb.ph; color: cb.label ? th.ink : th.ink4 }
        Icon { id: chev; name: "down"; color: th.ink3; width: 13; height: 13; anchors.right: parent.right; anchors.rightMargin: 9; anchors.verticalCenter: parent.verticalCenter }
        MouseArea { id: cma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: menu.openFor(cb, cb.options, cb.value, (v) => cb.picked(v)) }
    }

    component Tile: Rectangle {
        id: tl
        property string label: ""
        property string value: ""
        property string unit: ""
        property string note: ""
        property string tip: ""
        property real big: th.fs + 4
        radius: th.r2; color: th.fill
        Layout.fillHeight: true
        implicitHeight: tcol.implicitHeight + 2 * Math.round(10 * th.d); implicitWidth: 150
        Column {
            id: tcol; x: 12; y: Math.round(10 * th.d); width: tl.width - 24; spacing: 2
            Small { text: tl.label; width: parent.width; font.pixelSize: th.fs - 2 }
            Row {
                spacing: 4; width: parent.width
                Txt { id: tvv; text: tl.value; font.pixelSize: tl.big; font.weight: th.wth; elide: Text.ElideRight; width: Math.min(implicitWidth, parent.width - (tun.visible ? tun.implicitWidth + 4 : 0)) }
                Small { id: tun; visible: tl.unit !== ""; text: tl.unit; anchors.baseline: tvv.baseline }
            }
            Small { visible: tl.note !== ""; text: tl.note; width: parent.width; font.pixelSize: th.fs - 2.5 }
        }
        Tip { text: tl.tip || (tl.label.length > 18 ? tl.label : "") }
    }

    component Bar: Rectangle {
        id: br
        property real frac: 0
        property color tint: th.accent
        implicitHeight: 4; radius: height / 2; color: th.fill2; clip: true
        Rectangle {
            visible: br.frac >= 0; height: parent.height; radius: height / 2; color: br.tint
            width: Math.max(br.frac > 0 ? height : 0, Math.min(1, Math.max(0, br.frac)) * br.width)
            Behavior on width { enabled: th.motion; NumberAnimation { duration: 280; easing.type: Easing.OutCubic } }
        }
        Rectangle {
            id: ind; visible: br.frac < 0; height: parent.height; radius: height / 2; color: br.tint; width: br.width * .28
            NumberAnimation on x { running: br.frac < 0 && br.visible && th.motion; from: -br.width * .28; to: br.width; duration: 1400; loops: Animation.Infinite; easing.type: Easing.InOutSine }
        }
    }

    component Lamp: Icon {
        id: lmp
        property string st: "ok"
        property real spin: 0                       // ар1
        name: st === "ok" ? "ok" : st === "need" ? "need" : st === "bad" ? "bad" : st === "warn" ? "warn" : st === "wait" ? "wait" : "info"
        color: st === "ok" ? th.ok : st === "bad" ? th.bad : (st === "warn" || st === "need") ? th.warn : st === "wait" ? th.accent : th.ink3
        width: 16; height: 16
        rotation: st === "wait" ? spin : 0
        NumberAnimation on spin { running: lmp.st === "wait" && th.motion; from: 0; to: 360; duration: 1100; loops: Animation.Infinite }
    }

    component Pane: Rectangle {
        id: pn
        default property alias content: pbody.data
        property alias spacing: pbody.spacing
        property real padH: Math.round(20 * th.d)
        property real padV: Math.round(18 * th.d)
        property bool clickable: false
        readonly property bool hovered: pnm.containsMouse
        signal clicked()
        radius: th.r; color: th.card; border.width: th.hair; border.color: clickable && hovered ? th.edge2 : th.edge
        implicitHeight: pbody.implicitHeight + 2 * padV
        implicitWidth: pbody.implicitWidth + 2 * padH
        MouseArea { id: pnm; anchors.fill: parent; enabled: pn.clickable; hoverEnabled: pn.clickable; cursorShape: Qt.PointingHandCursor; onClicked: pn.clicked() }
        ColumnLayout { id: pbody; x: pn.padH; y: pn.padV; width: pn.width - 2 * pn.padH; spacing: Math.round(12 * th.d) }
    }

    component Collapse: ColumnLayout {
        id: cl
        property string title: ""
        property bool open: false
        default property alias content: cbody.data
        spacing: Math.round(10 * th.d)
        Item {
            Layout.fillWidth: true; implicitHeight: 24
            Row {
                anchors.verticalCenter: parent.verticalCenter; spacing: 6
                Txt { text: cl.title; color: hca.containsMouse ? th.ink : th.ink2; font.weight: Math.min(th.wt + 100, 600) }
                Icon { name: cl.open ? "up" : "down"; color: th.ink3; width: 12; height: 12; anchors.verticalCenter: parent.verticalCenter }
            }
            MouseArea { id: hca; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: cl.open = !cl.open }
        }
        ColumnLayout { id: cbody; Layout.fillWidth: true; visible: cl.open; spacing: Math.round(10 * th.d) }
    }

    component Hr: Rectangle { Layout.fillWidth: true; implicitHeight: Math.max(1, th.hair); color: th.sep }

    component Scroll: Item {
        id: sc
        default property alias content: scol.data
        property alias spacing: scol.spacing
        property alias flick: fl
        property real maxW: 0
        property real padB: th.pad
        clip: true
        function scrollBy(d) { fl.contentY = Math.max(0, Math.min(Math.max(0, fl.contentHeight - fl.height), fl.contentY + d)) }
        function toTop() { fl.contentY = 0 }
        Flickable {
            id: fl; anchors.fill: parent
            contentWidth: width; contentHeight: scol.implicitHeight + sc.padB
            interactive: false; boundsBehavior: Flickable.StopAtBounds
            Behavior on contentY { enabled: th.motion && !thumbMa.pressed; NumberAnimation { duration: 140; easing.type: Easing.OutCubic } }
            ColumnLayout {
                id: scol
                width: sc.maxW > 0 ? Math.min(sc.maxW, fl.width - 14) : fl.width - 14
                x: Math.max(0, (fl.width - 14 - width) / 2)
                spacing: th.gap
            }
        }
        WheelHandler {
            onWheel: (ev) => { tipLayer.hideNow(); sc.scrollBy(-(ev.pixelDelta.y !== 0 ? ev.pixelDelta.y : ev.angleDelta.y / 120 * 90)) }
        }
        Item {
            anchors.right: parent.right; anchors.top: parent.top; anchors.bottom: parent.bottom; width: 10
            visible: fl.contentHeight > fl.height + 1
            Rectangle {
                id: thumb; x: 2; width: thumbMa.containsMouse || thumbMa.pressed ? 7 : 5; radius: width / 2
                y: fl.visibleArea.yPosition * parent.height
                height: Math.max(28, fl.visibleArea.heightRatio * parent.height)
                color: thumbMa.pressed ? th.ink3 : thumbMa.containsMouse ? th.ink4 : th.fill3
            }
            MouseArea {
                id: thumbMa; anchors.fill: parent; hoverEnabled: true
                property real y0: 0
                property real c0: 0
                onPressed: (e) => { y0 = e.y; c0 = fl.contentY; if (e.y < thumb.y || e.y > thumb.y + thumb.height) { fl.contentY = Math.max(0, Math.min(fl.contentHeight - fl.height, (e.y / height) * fl.contentHeight - fl.height / 2)); c0 = fl.contentY } }
                onPositionChanged: (e) => { if (pressed) fl.contentY = Math.max(0, Math.min(fl.contentHeight - fl.height, c0 + (e.y - y0) / height * fl.contentHeight)) }
            }
        }
    }

    // ах1
    component Plot: Chart {
        rev: win.rev
        light: th.light; accent: th.accent; lw: ap.lw; fill: !!ap.cfill; grid: !!ap.cgrid; dots: !!ap.cdots
        family: th.ff; px: th.fs
    }

    component Legend: Flow {
        id: kf
        property var items: []
        spacing: 12
        Repeater {
            model: kf.items
            delegate: Row {
                required property var modelData
                spacing: 6
                Rectangle {
                    anchors.verticalCenter: parent.verticalCenter
                    width: modelData.dots ? 7 : 12; height: modelData.dots ? 7 : 3; radius: modelData.dots ? 3.5 : 1.5; color: th.sc(modelData.color)
                }
                Small { text: modelData.name; color: th.ink2; font.pixelSize: th.fs - 2 }
                Small { visible: !!modelData.value; text: modelData.value || ""; color: th.ink; font.weight: th.wth; font.pixelSize: th.fs - 2 }
            }
        }
    }

    // ач1
    Item {
        id: frame
        anchors.fill: parent
        anchors.margins: win.flat && ap.full === "gap" ? 7 : 0
        readonly property real wr: win.flat && ap.full !== "gap" ? 0 : 12

        Wall {
            id: wall; anchors.fill: parent; radius: frame.wr
            preset: ap.bg; light: th.light; solid: ap.solid || "#0e0f13"; img: ap.bg === "image" ? bgPath : ""; dim: ap.dim; blur: ap.iblur
        }

        RowLayout {
            id: appRow
            anchors.fill: parent; anchors.margins: th.pad
            spacing: th.pad
            layoutDirection: ap.side === "right" ? Qt.RightToLeft : Qt.LeftToRight

            Side { Layout.fillHeight: true; Layout.preferredWidth: win.compact ? 62 : Math.max(200, Math.min(340, ap.sidew)) }

            ColumnLayout {
                Layout.fillWidth: true; Layout.fillHeight: true; spacing: th.pad
                LayoutMirroring.enabled: false
                TopBar { Layout.fillWidth: true }
                Item {
                    id: views
                    Layout.fillWidth: true; Layout.fillHeight: true
                    SetupView { anchors.fill: parent; visible: view === "setup" }
                    TrainView { anchors.fill: parent; visible: view === "train" }
                    RunsView { id: runsView; anchors.fill: parent; visible: view === "runs" }
                    ChatView { id: chatView; anchors.fill: parent; visible: view === "chat" }
                    BenchView { id: benchView; anchors.fill: parent; visible: view === "bench" }
                    ModelsView { id: modelsView; anchors.fill: parent; visible: view === "models" }
                    SettingsView { anchors.fill: parent; visible: view === "settings" }
                }
            }
        }

        Rectangle { anchors.fill: parent; radius: frame.wr; color: "transparent"; border.width: frame.wr > 0 ? 1 : 0; border.color: th.light ? Qt.rgba(0, 0, 0, .14) : Qt.rgba(1, 1, 1, .1) }
    }

    // аш1
    Repeater {
        model: win.flat ? [] : [["l", Qt.LeftEdge, Qt.SizeHorCursor], ["r", Qt.RightEdge, Qt.SizeHorCursor], ["t", Qt.TopEdge, Qt.SizeVerCursor], ["b", Qt.BottomEdge, Qt.SizeVerCursor],
                               ["tl", Qt.TopEdge | Qt.LeftEdge, Qt.SizeFDiagCursor], ["tr", Qt.TopEdge | Qt.RightEdge, Qt.SizeBDiagCursor],
                               ["bl", Qt.BottomEdge | Qt.LeftEdge, Qt.SizeBDiagCursor], ["br", Qt.BottomEdge | Qt.RightEdge, Qt.SizeFDiagCursor]]
        delegate: MouseArea {
            required property var modelData
            readonly property string k: modelData[0]
            readonly property int g: 6
            x: k.indexOf("r") >= 0 ? win.width - g : 0
            y: k.indexOf("b") >= 0 ? win.height - g : 0
            width: k === "t" || k === "b" ? win.width : g * (k.length === 2 ? 2 : 1)
            height: k === "l" || k === "r" ? win.height : g * (k.length === 2 ? 2 : 1)
            z: 900; cursorShape: modelData[2]; hoverEnabled: true
            onPressed: win.startSystemResize(modelData[1])
        }
    }

    // аэ1
    component NavItem: Rectangle {
        id: ni
        property string go: ""
        property string icon: ""
        property string label: ""
        property string count: ""
        property string dot: ""          // аэ2
        property string dotTip: ""
        readonly property bool cur: view === go
        Layout.fillWidth: true
        implicitHeight: Math.round(30 * th.d + 2)
        radius: th.rc
        color: cur ? th.fill2 : nma.containsMouse ? th.fill : "transparent"
        RowLayout {
            anchors.fill: parent; anchors.leftMargin: win.compact ? 0 : 10; anchors.rightMargin: win.compact ? 0 : 10; spacing: 10
            Item { Layout.fillWidth: win.compact; visible: win.compact }
            Icon { name: ni.icon; color: ni.cur ? th.accent : th.ink3; Layout.preferredWidth: 17; Layout.preferredHeight: 17 }
            Txt { visible: !win.compact; text: ni.label; Layout.fillWidth: true; color: ni.cur || nma.containsMouse ? th.ink : th.ink2; font.weight: Math.min(th.wt + 60, 600) }
            Small { visible: !win.compact && ni.count !== ""; text: ni.count; font.pixelSize: th.fs - 2.5 }
            Rectangle {
                visible: ni.dot !== "" && !win.compact; Layout.preferredWidth: 7; Layout.preferredHeight: 7; radius: 3.5
                color: ni.dot === "ok" ? th.ok : ni.dot === "bad" ? th.bad : ni.dot === "need" ? th.warn : th.accent
                SequentialAnimation on opacity {
                    running: ni.dot === "live" && th.motion; loops: Animation.Infinite
                    NumberAnimation { to: .3; duration: 1000; easing.type: Easing.InOutSine }
                    NumberAnimation { to: 1; duration: 1000; easing.type: Easing.InOutSine }
                }
                Tip { text: ni.dotTip }
            }
            Item { Layout.fillWidth: win.compact; visible: win.compact }
        }
        MouseArea { id: nma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: win.go(ni.go) }
        Tip { text: win.compact ? ni.label : "" }
    }

    component Side: Rectangle {
        radius: th.r; color: th.mat; border.width: th.hair; border.color: th.edge
        ColumnLayout {
            anchors.fill: parent; anchors.topMargin: Math.round(14 * th.d); anchors.bottomMargin: Math.round(10 * th.d)
            anchors.leftMargin: Math.round(10 * th.d); anchors.rightMargin: Math.round(10 * th.d)
            spacing: 2
            Item {
                id: brand
                Layout.fillWidth: true; implicitHeight: 30; Layout.bottomMargin: Math.round(10 * th.d)
                RowLayout {
                    anchors.fill: parent; anchors.leftMargin: win.compact ? 0 : 8; anchors.rightMargin: win.compact ? 0 : 4; spacing: 10
                    Item { Layout.fillWidth: true; visible: win.compact }
                    Logo { ink: th.ink; accent: th.accent; Layout.preferredWidth: 26; Layout.preferredHeight: 26 }
                    Txt { visible: !win.compact; text: "TierLM"; font.pixelSize: th.fs + 2.5; font.weight: th.wth; Layout.fillWidth: true }
                    Rectangle {
                        visible: !win.compact; radius: height / 2; color: "transparent"; border.width: 1; border.color: th.edge2
                        implicitHeight: 18; implicitWidth: vtx.implicitWidth + 14
                        Txt { id: vtx; anchors.centerIn: parent; text: APPVER; color: th.ink3; font.pixelSize: th.fs - 3 }
                    }
                    Item { Layout.fillWidth: true; visible: win.compact }
                }
                DragHandler { target: null; onActiveChanged: if (active) win.startSystemMove() }
            }
            // аю1
            Rectangle {
                id: liveCard
                visible: liveName !== "" && !win.compact
                Layout.fillWidth: true; Layout.bottomMargin: Math.round(10 * th.d)
                implicitHeight: lcol.implicitHeight + 20
                radius: th.r2; color: lma.containsMouse ? th.wash2 : th.wash; border.width: 1; border.color: th.wash2
                readonly property var lt: tv && tv.name === liveName ? tv : null
                readonly property real lf: lt && lt.total ? Math.min(1, lt.step / lt.total) : liveRow && liveRow["всего"] && liveRow["шаг"] ? Math.min(1, liveRow["шаг"] / liveRow["всего"]) : -1
                ColumnLayout {
                    id: lcol; x: 12; y: 10; width: parent.width - 24; spacing: 7
                    RowLayout {
                        spacing: 7; Layout.fillWidth: true
                        Rectangle {
                            Layout.preferredWidth: 7; Layout.preferredHeight: 7; radius: 3.5; color: th.accent
                            SequentialAnimation on opacity { running: th.motion; loops: Animation.Infinite; NumberAnimation { to: .3; duration: 1000 } NumberAnimation { to: 1; duration: 1000 } }
                        }
                        Txt { text: liveName; Layout.fillWidth: true; font.weight: th.wth }
                        Txt { visible: liveCard.lf >= 0; text: Math.round(100 * liveCard.lf) + " %"; color: th.accentInk; font.pixelSize: th.fs - 1 }
                    }
                    Bar { Layout.fillWidth: true; frac: liveCard.lf }
                    Small {
                        Layout.fillWidth: true; color: th.ink2
                        text: liveCard.lt ? (liveCard.lt.bar && liveCard.lt.bar.cap ? liveCard.lt.bar.cap : liveCard.lt.title) : liveRow && liveRow["шаг"] ? tx("Шаг") + " " + liveRow["шаг"] + (liveRow["всего"] ? " " + tx("из") + " " + liveRow["всего"] : "") : tx("Выполняется")
                    }
                }
                MouseArea { id: lma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { B.openRun(liveName); win.go("train") } }
            }
            Small { visible: !win.compact; text: tx("Обучение"); font.weight: th.wth; Layout.leftMargin: 10; Layout.topMargin: 4; Layout.bottomMargin: 4; font.pixelSize: th.fs - 2.5 }
            NavItem { go: "setup"; icon: "setup"; label: tx("Новое обучение"); dot: setupDot(); dotTip: sv && sv.blocked && sv.blocked.length ? sv.blocked[0] : tx("Готово к запуску") }
            NavItem { go: "train"; icon: "train"; label: tx("Ход обучения"); dot: liveName ? "live" : ""; dotTip: liveName ? tx("Выполняется") + " «" + liveName + "»" : "" }
            NavItem { go: "runs"; icon: "runs"; label: tx("Запуски"); count: hh && hh["запуски"] && hh["запуски"].length ? String(hh["запуски"].length) : "" }
            Small { visible: !win.compact; text: tx("Модель"); font.weight: th.wth; Layout.leftMargin: 10; Layout.topMargin: 12; Layout.bottomMargin: 4; font.pixelSize: th.fs - 2.5 }
            NavItem { go: "models"; icon: "chip"; label: tx("Модели"); dot: hh && hh["настройка"] && hh["настройка_что"] === "веса" ? "live" : "" }
            NavItem { go: "chat"; icon: "chat"; label: tx("Разговор"); dot: chatDot() }
            NavItem { go: "bench"; icon: "bench"; label: tx("Проверка"); dot: bn && bn.run ? "live" : "" }
            Item { Layout.fillHeight: true }
            ColumnLayout {
                visible: !win.compact && !!(hh && hh["диск"] && hh["диск"]["всего"])
                Layout.fillWidth: true; Layout.leftMargin: 10; Layout.rightMargin: 10; Layout.bottomMargin: 6; spacing: 5
                readonly property real used: hh && hh["диск"] && hh["диск"]["всего"] ? 1 - hh["диск"]["свободно"] / hh["диск"]["всего"] : 0
                Small { Layout.fillWidth: true; font.pixelSize: th.fs - 2.5; text: hh && hh["диск"] ? tx("Свободно") + " " + gibS(hh["диск"]["свободно"]) + " " + tx("из") + " " + gibS(hh["диск"]["всего"]) : "" }
                Bar { Layout.fillWidth: true; implicitHeight: 3; frac: parent.used; tint: parent.used > .9 ? th.bad : parent.used > .75 ? th.warn : th.accent }
            }
            NavItem { go: "settings"; icon: "settings"; label: tx("Настройки") }
            Hr { Layout.topMargin: 6; Layout.bottomMargin: 6 }
            GridLayout {
                Layout.fillWidth: true; columnSpacing: 4; rowSpacing: 4
                columns: win.compact ? 1 : 4
                Seg { visible: !win.compact; small: true; options: [["ru", "RU"], ["en", "EN"]]; value: B.lang(); onPicked: (v) => B.setLang(v) }
                Item { Layout.fillWidth: true; visible: !win.compact }
                Btn { kind: "quiet"; size: "sm"; icon: "theme"; tip: tx("Сменить тему") + " · Ctrl+Shift+L"; onClicked: apSet({theme: th.light ? "dark" : "light"}); Layout.alignment: Qt.AlignHCenter }
                Btn { kind: "danger"; size: "sm"; icon: "power"; tip: tx("Завершить работу программы") + " · Ctrl+Q"; onClicked: killAll(); Layout.alignment: Qt.AlignHCenter }
            }
        }
    }
    function gibS(b) { if (!b && b !== 0) return "—"; const g = b / 1073741824; return (g >= 100 ? Math.round(g) : g.toFixed(1)).toString().replace(".", ui.dec === "comma" ? "," : ".") + " " + tx("ГиБ") }
    function setupDot() { if (liveName) return ""; if (!sv || !sv.launch) return ""; const p = sv.launch.pill; return p === "ok" ? "ok" : p === "bad" ? "bad" : p === "wait" ? "live" : "need" }
    function chatDot() { const r = hh ? hh["разговор"] : null; if (!r || r["имя"] === null || r["имя"] === undefined) return ""; return r["готов"] ? "ok" : "live" }

    // бв1
    component TopBar: Rectangle {
        id: tb
        implicitHeight: Math.round(44 * th.d + 6)
        radius: th.r; color: th.mat; border.width: th.hair; border.color: th.edge
        DragHandler { target: null; onActiveChanged: if (active) win.startSystemMove() }
        TapHandler { onDoubleTapped: win.toggleMax() }
        RowLayout {
            anchors.fill: parent; anchors.leftMargin: Math.round(16 * th.d); anchors.rightMargin: 8; spacing: 8
            H2 { text: viewTitle(); Layout.maximumWidth: implicitWidth }
            // бв2
            Seg {
                visible: view === "setup"; Layout.leftMargin: 6; Layout.rightMargin: 4
                options: [["tier", "TierLM"], ["lora", "LoRA"]]; value: sv && sv.tab ? sv.tab : "tier"
                onPicked: (v) => B.setTab(String(v))
            }
            Txt { text: viewSub(); color: th.ink3; font.pixelSize: th.fs - 1; Layout.fillWidth: true; Layout.minimumWidth: 0 }
            // бв3
            Row {
                spacing: 6; visible: view === "runs"
                Btn { size: "sm"; icon: "compare"; text: tx("Сравнить") + (runSelCount() >= 2 ? " · " + runSelCount() : ""); on: runSelCount() >= 2; tip: tx("Требуется выбрать от 2 до 6 запусков."); onClicked: B.compare(runSelNames()) }
                Btn { size: "sm"; kind: "danger"; icon: "trash"; text: tx("Удалить") + (runSelCount() ? " · " + runSelCount() : ""); on: runSelCount() > 0; onClicked: askDelete(runSelNames()) }
            }
            Row {
                spacing: 6; visible: view === "train" && !!(tv && tv.name)
                Btn { size: "sm"; kind: "quiet"; icon: "file"; text: tx("Журнал"); tip: "J"; onClicked: openWin("log") }
                Btn { size: "sm"; kind: "quiet"; icon: "settings"; text: tx("Параметры"); onClicked: openWin("plan") }
            }
            Btn {
                visible: view === "setup" && setupNarrow; size: "sm"; kind: "primary"; icon: "play"; text: tx("Начать обучение")
                on: !!(sv && sv.launch && sv.launch.can); tip: "Ctrl+Enter"; onClicked: B.launch()
            }
            Btn { kind: "quiet"; icon: "search"; tip: tx("Команды") + " · Ctrl+K"; onClicked: cmdp.open() }
            Rectangle { Layout.preferredWidth: 1; Layout.preferredHeight: 22; color: th.sep }
            Row {
                spacing: 2
                Btn { kind: "quiet"; size: "sm"; icon: "min"; tip: tx("Свернуть"); onClicked: win.showMinimized() }
                Btn { kind: "quiet"; size: "sm"; icon: "max"; tip: win.flat ? tx("Обычный размер") : tx("Развернуть"); onClicked: win.toggleMax() }
                Btn { id: xbtn; kind: "quiet"; size: "sm"; icon: "x"; tip: tx("Закрыть"); onClicked: win.close() }
            }
        }
        // бв4
        Item {
            anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom
            anchors.leftMargin: th.r * .7; anchors.rightMargin: th.r * .7; height: 2; clip: true
            visible: !!liveCard2.frac
            Rectangle { id: liveCard2; readonly property real frac: tv && tv.live && tv.total ? Math.min(1, tv.step / tv.total) : 0
                        height: 2; radius: 1; color: th.accent; width: parent.width * frac
                        Behavior on width { enabled: th.motion; NumberAnimation { duration: 600; easing.type: Easing.OutCubic } } }
        }
    }
    readonly property bool setupNarrow: views.width < 940
    function viewTitle() { return ({setup: tx("Новое обучение"), train: tx("Ход обучения"), runs: tx("Запуски"), chat: tx("Разговор"), bench: tx("Проверка"), models: tx("Модели"), settings: tx("Настройки")})[view] || "" }
    function viewSub() {
        if (view === "setup") return sv ? sv.sub || "" : ""
        if (view === "train") return tv && tv.name ? "«" + tv.name + "»" : ""
        if (view === "settings") { for (let i = 0; i < secs.length; i++) if (secs[i][0] === sec) return secs[i][1]; return "" }
        if (view === "chat") { const r = hh ? hh["разговор"] : null; return r && r["имя"] !== null && r["имя"] !== undefined ? (r["готов"] ? tx("Модель в видеопамяти") : tx("Загрузка весов")) : "" }
        return ""
    }
    function toggleMax() { if (win.visibility === Window.Maximized || win.visibility === Window.FullScreen) win.showNormal(); else win.showMaximized() }
    function go(v) {
        if (v === "train" && liveName && !B.current()) B.openRun(liveName)
        view = v
        tipLayer.hideNow()
        if (v === "chat") chatView.enter()
        if (v === "bench") benchView.enter()
        if (v === "models") modelsView.enter()
    }
    onClosing: (close) => {
        if (B.hideToTray()) { close.accepted = false; win.hide(); B.trayHint(); return }
        Qt.quit()
    }

    // би1
    component OptCard: Rectangle {
        id: oc
        property string name: ""
        property var chips: []
        property bool on: false
        property string tip: ""
        signal picked()
        radius: th.r2
        color: on ? th.wash : oma.containsMouse ? th.fill2 : th.fill
        border.width: on ? 1.5 : 0; border.color: th.accent
        implicitHeight: ocol.implicitHeight + 2 * Math.round(12 * th.d)
        ColumnLayout {
            id: ocol; x: 14; y: Math.round(12 * th.d); width: oc.width - 28; spacing: 7
            RowLayout {
                Layout.fillWidth: true
                Txt { text: oc.name; font.weight: th.wth; font.pixelSize: th.fs + .5; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone }
                Rectangle {
                    Layout.preferredWidth: 16; Layout.preferredHeight: 16; radius: 8; color: "transparent"; border.width: oc.on ? 5 : 1.5; border.color: oc.on ? th.accent : th.ink4
                    Layout.alignment: Qt.AlignTop
                }
            }
            Flow {
                Layout.fillWidth: true; spacing: 5
                Repeater { model: oc.chips; delegate: Chip { required property string modelData; text: modelData; tint: oc.on ? th.wash : th.fill; fg: oc.on ? th.accentInk : th.ink2; implicitHeight: 20 } }
            }
        }
        MouseArea { id: oma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: oc.picked() }
        Tip { text: oc.tip }
    }

    component CheckRow: Item {
        id: cr
        property var c: ({})
        Layout.fillWidth: true
        implicitHeight: crl.implicitHeight + 14
        ColumnLayout {
            id: crl; x: 0; y: 7; width: cr.width; spacing: 6
            RowLayout {
                Layout.fillWidth: true; spacing: 10
                Lamp { st: cr.c.s || "info" }
                Txt { text: cr.c.t || ""; color: th.ink2; Layout.preferredWidth: Math.round(cr.width * .34) }
                Txt { id: crv; text: cr.c.v || ""; Layout.fillWidth: true; color: cr.c.s === "bad" ? th.bad : th.ink; Tip { text: cr.c.sub || (crv.truncated ? crv.text : "") } }
            }
            Flow {
                visible: !!(cr.c.acts && cr.c.acts.length); Layout.fillWidth: true; Layout.leftMargin: 26; spacing: 6
                Repeater { model: (cr.c.acts || []).length; delegate: Btn { required property int index; readonly property var modelData: (cr.c.acts || [])[index] || []; size: "sm"; kind: modelData[2] ? "tint" : ""; text: modelData[0]; onClicked: setupAct(modelData[1]) } }
            }
        }
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep; opacity: .8 }
    }
    function setupAct(a) {
        if (a === "папка") return pickModelFolder()
        if (a === "pick") return pickData()
        if (a === "prep") return B.setFile(B.choice().file)
        if (a === "train") { B.openRun(liveName); return go("train") }
        if (a === "reset") return B.resetOver()
        if (a.indexOf("open:") === 0) { B.openRun(a.slice(5)); return go("train") }
        if (a === "веса" && !(B.settings()["свои"]["HF_REPO"])) return go("models")
        if (a === "движок" || a === "веса" || a === "пакеты" || a === "склеить") return B.setupRun(a)
        if (a.indexOf("base:") === 0) return B.setBase(a.slice(5))
    }
    function pickData() { const p = B.pick("файл"); if (p) B.setFile(p) }
    function pickModelFolder() { const p = B.pick("папка"); if (p) { B.saveSettings({MODEL: p}); toast(tx("Папка модели"), p, "ok") } }

    component LaunchCard: Pane {
        id: lc
        readonly property var la: sv && sv.launch ? sv.launch : null
        RowLayout {
            Layout.fillWidth: true
            H2 { text: tx("Запуск") }
            Pill { kind: lc.la ? lc.la.pill : ""; text: lc.la ? lc.la.pillText : "" }
            Item { Layout.fillWidth: true }
        }
        Flow {
            visible: !!(lc.la && lc.la.bigs.length); Layout.fillWidth: true; spacing: Math.round(18 * th.d)
            Repeater {
                model: ((lc.la ? lc.la.bigs : []) || []).length
                delegate: Column {
                    required property int index
                    readonly property var modelData: ((lc.la ? lc.la.bigs : []) || [])[index] || []
                    spacing: 0
                    Txt { text: modelData[0]; font.pixelSize: th.fs + 9; font.weight: th.wth }
                    Small { text: modelData[1] }
                    Tip { text: modelData[2] }
                }
            }
        }
        Btn {
            Layout.fillWidth: true; size: "lg"; kind: "primary"; icon: "play"; text: B.lang() && lc.la && !lc.la.can && sv && sv.launch && !sv.launch.checks ? "" : tx("Начать обучение"); kbd: "Ctrl ↵"
            on: !!(lc.la && lc.la.can); onClicked: B.launch()
        }
        Txt { visible: !!(lc.la && lc.la.why); text: lc.la ? lc.la.why : ""; color: th.bad; font.pixelSize: th.fs - 1; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone }
        Txt { text: tx("Проверки"); font.weight: th.wth; font.pixelSize: th.fs - .5; Layout.topMargin: 6 }
        ColumnLayout {
            Layout.fillWidth: true; spacing: 0
            Repeater { model: ((lc.la ? lc.la.checks : []) || []).length; delegate: CheckRow { required property int index; readonly property var modelData: ((lc.la ? lc.la.checks : []) || [])[index] || []; c: modelData } }
        }
        Rectangle {
            visible: !!(jb && jb.log); Layout.fillWidth: true; radius: th.r2; color: th.field; border.width: 1; border.color: th.edge
            implicitHeight: Math.min(220, slog.implicitHeight + 20)
            clip: true
            Flickable {
                id: slf; anchors.fill: parent; anchors.margins: 10; contentHeight: slog.implicitHeight; interactive: false; clip: true
                contentY: Math.max(0, contentHeight - height)
                Mono { id: slog; width: slf.width; text: jb ? jb.log : "" }
            }
            WheelHandler { onWheel: (ev) => { slf.contentY = Math.max(0, Math.min(slf.contentHeight - slf.height, slf.contentY - ev.angleDelta.y / 2)) } }
            Btn { anchors.top: parent.top; anchors.right: parent.right; anchors.margins: 6; size: "sm"; kind: "tint"; icon: "copy"; tip: tx("Копировать вывод")
                  onClicked: { B.copy(jb ? jb.log : ""); toast(tx("Скопировано"), "", "ok") } }
        }
        RowLayout {
            Layout.fillWidth: true; Layout.topMargin: 6
            Txt { text: tx("План"); font.weight: th.wth; font.pixelSize: th.fs - .5; Layout.fillWidth: true }
            Btn {
                visible: !!(lc.la && lc.la.rows.length) && (!lc.la.expert || lc.la.ownCount > 0); size: "sm"; kind: "quiet"
                text: lc.la && lc.la.expert ? tx("Сбросить собственные значения") : tx("Изменить")
                tip: lc.la && lc.la.expert ? tx("Вернуть значения, подобранные автоподбором.") : tx("Режим эксперта: ручная настройка каждого параметра.")
                onClicked: lc.la.expert ? B.resetOver() : B.setMode("эксперт")
            }
        }
        Small { visible: !!(lc.la && lc.la.planEmpty); text: lc.la ? lc.la.planEmpty : "" }
        ColumnLayout {
            Layout.fillWidth: true; spacing: 0
            Repeater {
                model: ((lc.la ? lc.la.rows : []) || []).length
                delegate: Item {
                    id: prw
                    required property int index
                    readonly property var modelData: ((lc.la ? lc.la.rows : []) || [])[index] || []
                    Layout.fillWidth: true; implicitHeight: Math.round(34 * th.d)
                    RowLayout {
                        anchors.fill: parent; spacing: 10
                        Txt { text: prw.modelData.label; color: th.ink2; Layout.fillWidth: true; Tip { text: prw.modelData.tip } }
                        Txt { visible: !prw.modelData.edit; text: prw.modelData.value; Layout.maximumWidth: Math.round(prw.width * .58); horizontalAlignment: Text.AlignRight; Tip { text: prw.modelData.tip } }
                        Field {
                            visible: prw.modelData.edit && !prw.modelData.opts.length; Layout.preferredWidth: Math.min(190, prw.width * .5); implicitHeight: Math.round(26 + 2 * th.d)
                            text: prw.modelData.own; ph: prw.modelData.ph; halign: TextInput.AlignRight; bad: prw.modelData.bad
                            onEdited: (t) => planT.put(prw.modelData.k, t)
                        }
                        Combo {
                            visible: prw.modelData.edit && prw.modelData.opts.length > 0; Layout.preferredWidth: Math.min(190, prw.width * .5); implicitHeight: Math.round(26 + 2 * th.d)
                            options: prw.modelData.opts; value: prw.modelData.cur
                            onPicked: (v) => B.setOver(prw.modelData.k, String(v))
                        }
                    }
                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep; opacity: .8 }
                }
            }
        }
        Btn { visible: !!(lc.la && lc.la.expert); size: "sm"; kind: "quiet"; icon: "chip"; text: tx("Настройка оборудования"); tip: tx("Запас видеопамяти, бюджет среза строк, режим ×3 (FP16/BF16), опережающая подкачка."); onClicked: { sec = "hw"; go("settings") } }
    }
    Timer { id: planT; interval: 300; property var q: ({}); function put(k, v) { q[k] = v; restart() }
            onTriggered: { for (const k in q) B.setOver(k, q[k]); q = ({}) } }

    component SetupView: Item {
        RowLayout {
            anchors.fill: parent; spacing: th.gap
            Scroll {
                Layout.fillWidth: true; Layout.fillHeight: true
                ModelCard { Layout.fillWidth: true }
                Pane {
                    Layout.fillWidth: true
                    H2 { text: tx("Режим") }
                    GridLayout {
                        Layout.fillWidth: true; columns: width > 1000 ? 4 : 2; columnSpacing: Math.round(8 * th.d); rowSpacing: Math.round(8 * th.d)
                        Repeater { model: ((sv ? sv.modes : []) || []).length; delegate: OptCard { required property int index; readonly property var modelData: ((sv ? sv.modes : []) || [])[index] || []; Layout.fillWidth: true; Layout.fillHeight: true; name: modelData.name; chips: modelData.chips; on: modelData.on; tip: modelData.tip; onPicked: B.setMode(modelData.k) } }
                    }
                }
                Pane {
                    Layout.fillWidth: true; visible: !(sv && sv.tab === "lora")
                    H2 { text: tx("Объект обучения") }
                    GridLayout {
                        Layout.fillWidth: true; columns: width > 620 ? 3 : 1; columnSpacing: Math.round(8 * th.d); rowSpacing: Math.round(8 * th.d)
                        Repeater { model: ((sv ? sv.goals : []) || []).length; delegate: OptCard { required property int index; readonly property var modelData: ((sv ? sv.goals : []) || [])[index] || []; Layout.fillWidth: true; Layout.fillHeight: true; name: modelData.name; chips: modelData.chips; on: modelData.on; tip: modelData.tip; onPicked: B.setGoal(modelData.k) } }
                    }
                }
                LoraCard { Layout.fillWidth: true; visible: !!(sv && sv.tab === "lora") }
                DataCard { Layout.fillWidth: true }
                LaunchCard { Layout.fillWidth: true; visible: setupNarrow }
            }
            Scroll { visible: !setupNarrow; Layout.preferredWidth: Math.max(360, Math.min(430, views.width * .34)); Layout.fillHeight: true; LaunchCard { Layout.fillWidth: true } }
        }
    }

    component ModelCard: Pane {
        id: mc
        readonly property var m: sv ? sv.model : null
        RowLayout {
            Layout.fillWidth: true
            H2 { text: tx("Модель") }
            Pill { kind: mc.m ? mc.m.pill : ""; text: mc.m ? mc.m.pillText : "" }
            Item { Layout.fillWidth: true }
            Btn { visible: !!(mc.m && mc.m["есть"]); size: "sm"; kind: "quiet"; icon: "folder"; text: tx("Изменить папку"); onClicked: pickModelFolder() }
        }
        ColumnLayout {
            visible: !!(mc.m && mc.m["есть"]); Layout.fillWidth: true; spacing: 4
            Txt { text: mc.m ? mc.m.name : ""; font.pixelSize: th.fs + 9; font.weight: th.wth; Layout.fillWidth: true; Tip { text: mc.m && mc.m.arch ? mc.m.arch : "" } }
            Txt { text: mc.m ? mc.m.line : ""; color: th.ink2; Layout.fillWidth: true }
            Flow { Layout.fillWidth: true; Layout.topMargin: 8; spacing: 5; Repeater { model: mc.m ? mc.m.chips : []; delegate: Chip { required property string modelData; text: modelData } } }
            RowLayout {
                Layout.fillWidth: true; Layout.topMargin: 8; spacing: 6
                Icon { name: "lock"; color: th.ink3; Layout.preferredWidth: 14; Layout.preferredHeight: 14 }
                Small { id: mpath; text: mc.m ? mc.m.path : ""; Layout.fillWidth: true; elide: Text.ElideMiddle; Tip { text: mpath.truncated ? mpath.text : "" } }
                Chip { text: tx("Только чтение"); implicitHeight: 19; tip: tx("Файл используется только для чтения.") }
            }
            Rectangle {
                visible: !!(mc.m && mc.m.why); Layout.fillWidth: true; Layout.topMargin: 8; radius: th.r2; color: th.badW
                implicitHeight: wtx.implicitHeight + 18
                RowLayout { anchors.fill: parent; anchors.margins: 9; spacing: 8
                    Icon { name: "bad"; color: th.bad; Layout.preferredWidth: 16; Layout.preferredHeight: 16; Layout.alignment: Qt.AlignTop }
                    Txt { id: wtx; text: mc.m ? mc.m.why : ""; color: th.ink; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone } }
            }
            BaseList { visible: !!(mc.m && mc.m.base); Layout.fillWidth: true; Layout.topMargin: 12; bs: mc.m ? mc.m.base : null }
        }
        ColumnLayout {
            visible: !!(mc.m && !mc.m["есть"]); Layout.fillWidth: true; spacing: 10; Layout.topMargin: 10; Layout.bottomMargin: 6
            Icon { name: "folder"; color: th.ink4; Layout.preferredWidth: 34; Layout.preferredHeight: 34; Layout.alignment: Qt.AlignHCenter; sw: 1.2 }
            Txt { text: tx("Модель не найдена"); font.pixelSize: th.fs + 3; font.weight: th.wth; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
            Small { text: mc.m ? tx("В папке") + " " + mc.m.path + " " + tx("отсутствует model.safetensors.") : ""; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.WrapAnywhere; elide: Text.ElideNone }
            Row {
                Layout.alignment: Qt.AlignHCenter; spacing: 8
                Btn { kind: "primary"; icon: "folder"; text: tx("Указать папку"); onClicked: pickModelFolder() }
                Btn { icon: "save"; text: tx("Скачать"); onClicked: setupAct("веса") }
            }
        }
        Collapse {
            visible: !!(sv && sv.mem && sv.mem.length); Layout.fillWidth: true; title: tx("Память при обучении")
            GridLayout {
                Layout.fillWidth: true; columns: Math.max(1, Math.min(4, Math.floor(width / 170))); columnSpacing: Math.round(8 * th.d); rowSpacing: Math.round(8 * th.d)
                Repeater { model: ((sv ? sv.mem : []) || []).length; delegate: Tile { required property int index; readonly property var modelData: ((sv ? sv.mem : []) || [])[index] || []; Layout.fillWidth: true; label: modelData[0]; value: modelData[1]; note: modelData[2] } }
            }
        }
    }

    // бт1
    component BaseList: ColumnLayout {
        id: bl
        property var bs: null
        readonly property var rows: bs ? bs.rows : []
        spacing: 6
        RowLayout {
            Layout.fillWidth: true; spacing: 4
            Txt { text: tx("База"); font.weight: th.wth; font.pixelSize: th.fs - .5
                  Tip { text: tx("Замороженные матрицы хранятся в FP16 или в выбранном формате квантования. Обучаемые веса читаются из исходного BF16 и хранятся в FP32; замороженные нормировки и эмбеддинги преобразуются в FP16.") } }
            Item { Layout.fillWidth: true }
            Btn { id: qbtn; size: "sm"; kind: "quiet"; icon: "build"; text: tx("Квантовать"); on: !!(bl.bs && bl.bs.can && bl.bs.recipes.length)
                  tip: bl.bs && !bl.bs.can ? bl.bs.why : tx("Собственный квант из BF16, создаваемый на процессоре по правилам llama.cpp. Исходный файл используется только для чтения.")
                  onClicked: menu.openFor(qbtn, bl.bs.recipes, "", (v) => B.setupRun("квант:" + v)) }
            Btn { size: "sm"; kind: "quiet"; icon: "file"; text: tx("Из файла"); on: !!(bl.bs && bl.bs.can)
                  tip: bl.bs && !bl.bs.can ? bl.bs.why : tx("Готовый квант этой же модели: GGUF любого типа, в том числе динамический Unsloth, или bitsandbytes 4 бита. Блоки переносятся без пересчёта и сверяются с исходной моделью.")
                  onClicked: { const p = B.pick("квант"); if (p) B.setupRun("импорт:" + p) } }
        }
        Repeater {
            model: bl.rows.length
            delegate: Rectangle {
                id: br
                required property int index
                readonly property var r: bl.rows[index] || ({})
                Layout.fillWidth: true; implicitHeight: brl.implicitHeight + Math.round(14 * th.d)
                radius: th.rc
                color: r.on ? th.wash : brm.containsMouse && !r.bad ? th.fill2 : th.fill
                border.width: r.on ? 1.5 : 0; border.color: th.accent
                MouseArea { id: brm; anchors.fill: parent; hoverEnabled: true; cursorShape: br.r.bad || br.r.on ? Qt.ArrowCursor : Qt.PointingHandCursor
                            onClicked: if (!br.r.bad && !br.r.on) B.setBase(br.r.k) }
                Tip { text: br.r.tip || "" }
                RowLayout {
                    id: brl; x: 12; width: parent.width - 18; anchors.verticalCenter: parent.verticalCenter; spacing: 8
                    Rectangle { Layout.preferredWidth: 14; Layout.preferredHeight: 14; radius: 7; color: "transparent"; border.width: 1.5; border.color: br.r.on ? th.accent : th.edge2
                        Rectangle { anchors.centerIn: parent; width: 6; height: 6; radius: 3; color: th.accent; visible: !!br.r.on } }
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: 1
                        Txt { text: br.r.name || ""; Layout.fillWidth: true; color: br.r.on ? th.accentInk : th.ink; font.weight: br.r.on ? Math.min(th.wt + 150, 650) : th.wt }
                        Small { text: br.r.sub || ""; Layout.fillWidth: true }
                    }
                    Pill { kind: "bad"; text: br.r.bad || ""; tip: br.r.badTip || "" }
                    Btn { visible: !!br.r.del; size: "sm"; kind: "quiet"; icon: "trash"; tip: tx("Удалить файл базы")
                          onClicked: ask({title: tx("Удалить") + " «" + br.r.name + "»?", text: tx("Файл базы удаляется с диска. Исходная модель не затрагивается."), ok: tx("Удалить"), danger: true},
                                         (ok) => { if (ok) B.delBase(br.r.k) }) }
                }
            }
        }
        ColumnLayout {
            visible: !!(bl.bs && bl.bs.busy); Layout.fillWidth: true; spacing: 4; Layout.topMargin: 2
            Bar { Layout.fillWidth: true; frac: jb && jb.frac !== undefined ? jb.frac : -1 }
            Small { text: jb && jb.line ? jb.line : (bl.bs ? bl.bs.busy : ""); Layout.fillWidth: true; elide: Text.ElideMiddle }
        }
    }

    // бу1
    component Tog: Rectangle {
        id: tgc
        property string text: ""
        property string tip: ""
        property bool on: false
        signal toggled()
        implicitHeight: Math.round(26 + 2 * th.d); implicitWidth: tgt.implicitWidth + 24
        radius: th.rc
        color: on ? th.wash : tgm.containsMouse ? th.fill2 : th.fill
        border.width: on ? 1.5 : 0; border.color: th.accent
        Txt { id: tgt; anchors.centerIn: parent; text: tgc.text; color: tgc.on ? th.accentInk : th.ink2; font.pixelSize: th.fs - 1
              font.weight: tgc.on ? Math.min(th.wt + 150, 650) : th.wt }
        MouseArea { id: tgm; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: tgc.toggled() }
        Tip { text: tgc.tip }
    }

    // бф1
    component LoraCard: Pane {
        id: lrc
        readonly property var lo: sv && sv.lora ? sv.lora : null
        function syncAlpha() { lalpha.text = B.choice().alpha }
        RowLayout {
            Layout.fillWidth: true
            H2 { text: "LoRA"; Tip { text: tx("Классическая LoRA (Hu и др., 2021), как в PEFT: W + α/r · B·A поверх замороженных весов. A инициализируется равномерно в ±1/√вход, B — нулями, оптимизатор AdamW в FP32. Исходные веса не изменяются, адаптеры сохраняются отдельным файлом.") } }
            Pill { kind: "bad"; text: lrc.lo && lrc.lo.over ? tx("Больше бюджета") : ""; tip: lrc.lo ? lrc.lo.overText : "" }
            Item { Layout.fillWidth: true }
            Btn { visible: !!(lrc.lo && !lrc.lo.isDef); size: "sm"; kind: "quiet"; text: tx("По умолчанию"); tip: tx("Значения по умолчанию: r 16, α 32, все матрицы, dropout 0,05.")
                  onClicked: { B.setLora("def", ""); lrc.syncAlpha() } }
        }
        GridLayout {
            Layout.fillWidth: true; columns: width > 520 ? 3 : 1; columnSpacing: Math.round(22 * th.d); rowSpacing: Math.round(10 * th.d)
            ColumnLayout {
                spacing: 5; Layout.alignment: Qt.AlignTop
                Small { text: tx("Ранг r"); color: th.ink2; Tip { text: tx("Число направлений поправки в каждой матрице. Чем больше ранг, тем больше обучаемых чисел и расход видеопамяти.") } }
                Seg { options: lrc.lo ? lrc.lo.ranks : []; value: lrc.lo ? lrc.lo.r : ""; onPicked: (v) => B.setLora("r", String(v)) }
            }
            ColumnLayout {
                spacing: 5; Layout.alignment: Qt.AlignTop
                Small { text: "Alpha"; color: th.ink2; Tip { text: tx("Масштаб поправки: s = α / r. Обычно α = 2r.") } }
                Field { id: lalpha; Layout.preferredWidth: Math.round(96 * th.d); text: B.choice().alpha; halign: TextInput.AlignRight
                        onDone: (t) => { B.setLora("alpha", t); lrc.syncAlpha() }
                        input.validator: IntValidator { bottom: 1; top: 4096 } }
            }
            ColumnLayout {
                spacing: 5; Layout.alignment: Qt.AlignTop; Layout.fillWidth: true
                Small { text: "Dropout"; color: th.ink2; Tip { text: tx("Доля входа адаптера, обнуляемая при обучении. На валидации и в разговоре не применяется.") } }
                Seg { options: lrc.lo ? lrc.lo.drops : []; value: lrc.lo ? lrc.lo.drop : ""; onPicked: (v) => B.setLora("drop", String(v)) }
                Item { Layout.fillWidth: true; Layout.preferredHeight: 0 }    // бф2
            }
        }
        RowLayout {
            Layout.fillWidth: true; Layout.topMargin: 2; spacing: 8
            Small { text: tx("Матрицы"); color: th.ink2; Layout.fillWidth: true
                    Tip { text: tx("Матрицы, на которых устанавливаются адаптеры: q, k, v, o — внимание; gate, up, down — MLP; во всех слоях. В слоях полного внимания отдельной v_proj нет: V берётся из K.") } }
            Seg { small: true; options: lrc.lo ? lrc.lo.sets : []; value: lrc.lo ? lrc.lo.tg : ""; onPicked: (v) => B.setLora("tg", String(v)) }
        }
        Flow {
            Layout.fillWidth: true; spacing: 6
            Repeater {
                model: ((lrc.lo ? lrc.lo.mats : []) || []).length
                delegate: Tog {
                    required property int index
                    readonly property var modelData: ((lrc.lo ? lrc.lo.mats : []) || [])[index] || ({})
                    text: modelData.name || ""; on: !!modelData.on; tip: modelData.last ? tx("Должна остаться хотя бы одна матрица.") : ""
                    onToggled: B.setLora("mat", modelData.k)
                }
            }
        }
        GridLayout {
            Layout.fillWidth: true; Layout.topMargin: 4
            columns: Math.max(2, Math.min(4, Math.floor(width / 150))); columnSpacing: Math.round(8 * th.d); rowSpacing: Math.round(8 * th.d)
            Repeater { model: ((lrc.lo ? lrc.lo.tiles : []) || []).length; delegate: Tile { required property int index; readonly property var modelData: ((lrc.lo ? lrc.lo.tiles : []) || [])[index] || []; Layout.fillWidth: true; label: modelData[0]; value: modelData[1]; note: modelData[2] } }
        }
    }

    component DataCard: Pane {
        id: dc
        readonly property var dd: sv ? sv.data : null
        property int exAt: 0
        property var ex: null
        function loadEx() { if (dd && dd.path) ex = B.peek(dd.path, exAt) }
        Connections { target: B; function onMisc_changed() { const c = B.choice(); dfile.text = c.file; dname.text = c.name; dshare.text = c.share; dalw.text = c.always; dc.exAt = 0; dc.ex = null } }
        RowLayout {
            Layout.fillWidth: true
            H2 { text: tx("Данные") }
            Pill { kind: dc.dd ? dc.dd.pill : ""; text: dc.dd ? dc.dd.pillText : "" }
            Item { Layout.fillWidth: true }
        }
        Small { text: tx("Файл с примерами"); color: th.ink2; Tip { text: tx("Один пример на строку: {\"instruction\": \"…\", \"input\": \"…\", \"output\": \"…\"}.") } }
        RowLayout {
            Layout.fillWidth: true; spacing: 8
            Field { id: dfile; Layout.fillWidth: true; ph: tx("Путь к файлу .jsonl"); text: B.choice().file; onAccepted: B.setFile(text) }
            Btn { icon: "folder"; text: tx("Выбрать"); onClicked: pickData() }
        }
        Small { text: tx("Имя запуска"); color: th.ink2 }
        RowLayout {
            Layout.fillWidth: true; spacing: 8
            Field { id: dname; Layout.fillWidth: true; ph: tx("По имени файла"); text: B.choice().name; bad: !!(dc.dd && dc.dd.taken); onEdited: (t) => nameT.restart() }
            Pill { kind: "bad"; text: dc.dd ? dc.dd.taken : ""; tip: dc.dd ? dc.dd.takenTip : "" }
        }
        Timer { id: nameT; interval: 260; onTriggered: B.setName(dname.text) }
        // бх1
        ColumnLayout {
            visible: !!(dc.dd && dc.dd.state === "parsing"); Layout.fillWidth: true; spacing: 8; Layout.topMargin: 4
            RowLayout { Layout.fillWidth: true; Small { text: dc.dd ? dc.dd.cap || "" : ""; color: th.ink2; Layout.fillWidth: true } }
            Bar { Layout.fillWidth: true; frac: -1 }
            Mono { visible: !!(dc.dd && dc.dd.out); text: dc.dd ? dc.dd.out : ""; Layout.fillWidth: true }
        }
        Rectangle {
            visible: !!(dc.dd && (dc.dd.state === "error" || dc.dd.state === "stale")); Layout.fillWidth: true; radius: th.r2
            color: dc.dd && dc.dd.state === "stale" ? th.warnW : th.badW; implicitHeight: nrow.implicitHeight + 18
            RowLayout {
                id: nrow; anchors.fill: parent; anchors.margins: 9; spacing: 8
                Icon { name: dc.dd && dc.dd.state === "stale" ? "warn" : "bad"; color: dc.dd && dc.dd.state === "stale" ? th.warn : th.bad; Layout.preferredWidth: 16; Layout.preferredHeight: 16 }
                Txt { text: dc.dd ? (dc.dd.state === "stale" ? tx("Файл изменён после токенизации.") : dc.dd.err) : ""; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone }
                Btn { visible: !!(dc.dd && dc.dd.state === "stale"); size: "sm"; kind: "tint"; text: tx("Токенизировать заново"); onClicked: B.setFile(B.choice().file) }
            }
        }
        Mono { visible: !!(dc.dd && dc.dd.state === "error" && dc.dd.out); text: dc.dd ? dc.dd.out : ""; Layout.fillWidth: true }
        GridLayout {
            visible: !!(dc.dd && dc.dd.tiles.length); Layout.fillWidth: true; Layout.topMargin: 4
            columns: Math.max(2, Math.min(4, Math.floor(width / 150))); columnSpacing: Math.round(8 * th.d); rowSpacing: Math.round(8 * th.d)
            Repeater { model: ((dc.dd ? dc.dd.tiles : []) || []).length; delegate: Tile { required property int index; readonly property var modelData: ((dc.dd ? dc.dd.tiles : []) || [])[index] || []; Layout.fillWidth: true; label: modelData[0]; value: modelData[1]; unit: modelData[2] } }
        }
        // бх2
        GridLayout {
            visible: !!(dc.dd && dc.dd.instr > 0); Layout.fillWidth: true; Layout.topMargin: 4
            columns: width < 420 ? 1 : 2; columnSpacing: Math.round(10 * th.d); rowSpacing: Math.round(8 * th.d)
            ColumnLayout {
                spacing: 5; Layout.alignment: Qt.AlignTop
                Small { text: tx("С системной инструкцией, %"); color: th.ink2; Tip { text: tx("Доля примеров, в которых поле instruction передаётся как системная инструкция. В остальных примерах она отсутствует, благодаря чему манера ответов сохраняется и в обычном разговоре.") } }
                Field { id: dshare; Layout.preferredWidth: Math.round(96 * th.d); text: B.choice().share; halign: TextInput.AlignRight
                        onDone: (t) => B.setScene(t, dalw.text)
                        input.validator: IntValidator { bottom: 0; top: 100 } }
            }
            ColumnLayout {
                spacing: 5; Layout.fillWidth: true; Layout.alignment: Qt.AlignTop
                Small { text: tx("Всегда с системной инструкцией"); color: th.ink2; Tip { text: tx("Системная инструкция добавляется к примеру всегда, если в его поле instruction встречается одно из слов.") } }
                Field { id: dalw; Layout.fillWidth: true; text: B.choice().always; ph: tx("Слова из instruction, через запятую"); onDone: (t) => B.setScene(dshare.text, t) }
            }
        }
        Small { visible: !!(dc.dd && dc.dd.instr > 0 && dc.dd.state !== "parsing"); text: dc.dd ? dc.dd.sceneNote : ""; color: th.ink3 }
        Hr { visible: !!(dc.dd && dc.dd.path); Layout.topMargin: 4 }
        Collapse {
            id: exc; visible: !!(dc.dd && dc.dd.path); Layout.fillWidth: true; title: tx("Примеры из файла")
            onOpenChanged: if (open && !dc.ex) dc.loadEx()
            RowLayout {
                Layout.fillWidth: true
                Small { text: dc.ex && dc.ex.ok ? tx("Пример") + " " + (dc.ex.at + 1) + " " + tx("из") + " " + dc.ex.n : ""; Layout.fillWidth: true }
                Btn { size: "sm"; kind: "quiet"; icon: "left"; on: !!(dc.ex && dc.ex.ok && dc.ex.at > 0); tip: tx("Предыдущий"); onClicked: { dc.exAt = Math.max(0, dc.exAt - 1); dc.loadEx() } }
                Btn { size: "sm"; kind: "quiet"; icon: "right"; on: !!(dc.ex && dc.ex.ok && dc.ex.at < dc.ex.n - 1); tip: tx("Следующий"); onClicked: { dc.exAt = dc.exAt + 1; dc.loadEx() } }
            }
            Txt { visible: !!(dc.ex && !dc.ex.ok); text: dc.ex ? dc.ex.why || "" : ""; color: th.bad }
            Repeater {
                model: dc.ex && dc.ex.ok ? dc.ex.fields : []
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true; radius: th.r2; color: th.field; border.width: 1; border.color: th.edge
                    implicitHeight: fcol.implicitHeight + 18
                    ColumnLayout {
                        id: fcol; x: 11; y: 9; width: parent.width - 22; spacing: 3
                        Small { text: modelData[0]; font.pixelSize: th.fs - 2.5 }
                        TextEdit {
                            Layout.fillWidth: true; text: modelData[1].length > 4000 ? modelData[1].slice(0, 4000) + " …" : modelData[1]
                            readOnly: true; selectByMouse: true; wrapMode: TextEdit.Wrap; color: th.ink; font.family: th.ff; font.pixelSize: th.fs - .5
                            selectionColor: th.wash2; renderType: TextEdit.NativeRendering
                        }
                    }
                }
            }
        }
    }

    // бц1
    readonly property var chartOrder: { const a = (ui.charts || []).filter(k => ["loss", "acc", "vram", "temp", "time", "grad"].indexOf(k) >= 0); return a.length ? a : ["loss", "acc", "vram", "temp", "time", "grad"] }

    component PlotTile: Rectangle {
        id: pt
        property string k: ""
        property int n: 0
        readonly property var d: tv && tv.tiles ? tv.tiles[k] : null
        radius: th.r; color: th.card; border.width: th.hair; border.color: pma.containsMouse ? th.edge2 : th.edge
        implicitHeight: pcol.implicitHeight + 28
        ColumnLayout {
            id: pcol; x: 16; y: 14; width: pt.width - 32; spacing: 4
            RowLayout {
                Layout.fillWidth: true; spacing: 8
                Txt { text: pt.d ? pt.d.title : ""; color: th.ink2; font.pixelSize: th.fs - 1 }
                Rectangle { implicitWidth: 16; implicitHeight: 16; radius: 4; color: th.fill2; Txt { anchors.centerIn: parent; text: pt.n; color: th.ink3; font.pixelSize: th.fs - 3.5 } Tip { text: tx("Клавиша") + " " + pt.n } }
                Item { Layout.fillWidth: true }
                Row {
                    visible: !!(pt.d && pt.d.flag); spacing: 5
                    Icon { name: "warn"; color: th.warn; width: 14; height: 14; anchors.verticalCenter: parent.verticalCenter }
                    Small { text: pt.d ? pt.d.flag : ""; color: th.warn }
                }
                Icon { visible: !(pt.d && pt.d.flag); name: "expand"; color: pma.containsMouse ? th.ink2 : th.ink4; Layout.preferredWidth: 14; Layout.preferredHeight: 14 }
            }
            Row {
                spacing: 5; visible: !!(pt.d && pt.d.has)
                Txt { id: pv; text: pt.d ? pt.d.value : ""; font.pixelSize: th.fs + 11; font.weight: th.wth }
                Small { text: pt.d ? pt.d.unit : ""; anchors.baseline: pv.baseline; font.pixelSize: th.fs - 1 }
            }
            Legend { visible: !!(pt.d && pt.d.keys.length); Layout.fillWidth: true; items: pt.d ? pt.d.keys : [] }
            Small { visible: !!(pt.d && !pt.d.keys.length && pt.d.sub); text: pt.d ? pt.d.sub : ""; color: th.ink2 }
            Plot { Layout.fillWidth: true; Layout.preferredHeight: ui.ch || 150; key: "tile:" + pt.k; compact: true; empty: tv && tv.blank ? "" : tx("Данные появятся после первых шагов.") }
        }
        MouseArea { id: pma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; acceptedButtons: Qt.LeftButton; propagateComposedEvents: true; onClicked: openWin(pt.k) }
    }

    component StageCard: Rectangle {
        id: sgc
        property var s: ({})
        radius: th.r2
        color: s.cls === "cur" ? th.wash : th.fill
        border.width: s.cls === "cur" ? 1 : 0; border.color: th.wash2
        implicitHeight: scol2.implicitHeight + 20
        ColumnLayout {
            id: scol2; x: 11; y: 10; width: sgc.width - 22; spacing: 3
            RowLayout {
                spacing: 5; Layout.fillWidth: true
                Icon { name: sgc.s.icon || "todo"; width: 13; height: 13; Layout.preferredWidth: 13; Layout.preferredHeight: 13
                       color: sgc.s.cls === "done" ? th.ok : sgc.s.cls === "bad" ? th.bad : sgc.s.cls === "cur" ? th.accent : sgc.s.cls === "stop" ? th.warn : th.ink4 }
                Txt { text: sgc.s.name || ""; Layout.fillWidth: true; font.pixelSize: th.fs - 1; font.weight: th.wth; color: sgc.s.cls === "todo" || sgc.s.cls === "skip" ? th.ink3 : th.ink }
            }
            Small { id: scap; text: sgc.s.cap || " "; Layout.fillWidth: true; font.pixelSize: th.fs - 2.5; Tip { text: scap.truncated ? scap.text : "" } }
            Bar {
                Layout.fillWidth: true; Layout.topMargin: 5; implicitHeight: 3
                frac: sgc.s.cls === "todo" ? 0 : sgc.s.frac === undefined ? 0 : sgc.s.frac
                tint: sgc.s.cls === "done" ? th.ok : sgc.s.cls === "skip" ? th.fill3 : sgc.s.cls === "bad" ? th.bad : sgc.s.cls === "stop" ? th.warn : th.accent
            }
        }
    }

    component TrainView: Item {
        Pane {
            visible: !(hh && hh["запуски"] && hh["запуски"].length) && !(tv && tv.name)
            anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
            ColumnLayout {
                Layout.fillWidth: true; spacing: 10; Layout.topMargin: 26; Layout.bottomMargin: 26
                Icon { name: "train"; color: th.ink4; Layout.preferredWidth: 34; Layout.preferredHeight: 34; Layout.alignment: Qt.AlignHCenter; sw: 1.2 }
                Txt { text: tx("Обучений пока нет"); font.pixelSize: th.fs + 3; font.weight: th.wth; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
                Btn { kind: "primary"; text: tx("Новое обучение"); Layout.alignment: Qt.AlignHCenter; onClicked: go("setup") }
            }
        }
        Scroll {
            id: tscroll
            anchors.fill: parent
            visible: !!(tv && tv.name)
            maxW: ap.wide ? 0 : 1560
            Pane {
                Layout.fillWidth: true
                RowLayout {
                    id: thd
                    Layout.fillWidth: true; spacing: 10
                    readonly property bool tight: width < 720           // бэ1
                    Combo {
                        Layout.fillWidth: true; Layout.maximumWidth: 220; Layout.minimumWidth: 110
                        options: (hh && hh["запуски"] ? hh["запуски"] : []).map(r => [r["имя"], r["имя"]])
                        value: tv ? tv.name : ""; onPicked: (v) => B.openRun(v)
                    }
                    Pill { kind: tv ? tv.pill : ""; text: tv ? tv.word : "" }
                    Item { Layout.fillWidth: true }
                    Repeater {
                        model: ((tv ? tv.acts : []) || []).length
                        delegate: Btn {
                            required property int index
                            readonly property var modelData: ((tv ? tv.acts : []) || [])[index] || []
                            readonly property bool bare: thd.tight && modelData.kind !== "primary" && modelData.icon !== ""
                            kind: modelData.kind; icon: modelData.icon; text: bare ? "" : modelData.text; on: modelData.on
                            tip: [bare ? modelData.text : "", modelData.tip || ""].filter(x => x).join(" · ")
                            onClicked: trainAct(modelData.id)
                        }
                    }
                }
                Txt { text: tv ? tv.title : ""; font.pixelSize: th.fs + 9; font.weight: th.wth; Layout.fillWidth: true; Layout.topMargin: 2 }
                Rich { text: tv ? tv.sub : ""; color: th.ink2; Layout.fillWidth: true; wrapMode: Text.Wrap; visible: text !== "" }
                Rectangle {
                    visible: !!(tv && tv.err); Layout.fillWidth: true; radius: th.r2; color: th.badW; implicitHeight: erow.implicitHeight + 18
                    RowLayout { id: erow; anchors.fill: parent; anchors.margins: 9; spacing: 8
                        Icon { name: "bad"; color: th.bad; Layout.preferredWidth: 16; Layout.preferredHeight: 16; Layout.alignment: Qt.AlignTop }
                        Txt { text: tv ? tv.err : ""; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone } }
                }
                ColumnLayout {
                    visible: !!(tv && tv.bar && tv.bar.show); Layout.fillWidth: true; spacing: 7; Layout.topMargin: 6
                    RowLayout { Layout.fillWidth: true
                        Small { text: tv && tv.bar ? tv.bar.cap : ""; color: th.ink2; Layout.fillWidth: true }
                        Small { text: tv && tv.bar ? tv.bar.pct : ""; color: th.ink2 } }
                    Bar { Layout.fillWidth: true; implicitHeight: 5; frac: tv && tv.bar ? tv.bar.frac : 0; tint: tv && tv.bar && tv.bar.ok ? th.ok : th.accent }
                }
                GridLayout {
                    Layout.fillWidth: true; Layout.topMargin: 6
                    columns: width > 880 ? 7 : width > 500 ? 4 : 2; columnSpacing: Math.round(8 * th.d); rowSpacing: Math.round(8 * th.d)
                    Repeater { model: ((tv && tv.stages ? tv.stages : []) || []).length; delegate: StageCard { required property int index; readonly property var modelData: ((tv && tv.stages ? tv.stages : []) || [])[index] || []; Layout.fillWidth: true; s: modelData } }
                }
                ColumnLayout {
                    visible: !!(tv && tv.sweep && tv.sweep.show); Layout.fillWidth: true; spacing: 6; Layout.topMargin: 6
                    RowLayout { Layout.fillWidth: true
                        Txt { text: tv && tv.sweep ? tv.sweep.title : ""; font.weight: th.wth; font.pixelSize: th.fs - .5 }
                        Icon { name: "info"; color: th.ink4; Layout.preferredWidth: 14; Layout.preferredHeight: 14; Tip { text: tv && tv.sweep ? tv.sweep.note : "" } }
                        Item { Layout.fillWidth: true }
                        Rich { text: tv && tv.sweep ? tv.sweep.cap : ""; color: th.ink2; font.pixelSize: th.fs - 1 } }
                    Plot { Layout.fillWidth: true; Layout.preferredHeight: 190; key: "sweep" }
                }
                Hr { Layout.topMargin: 4 }
                Collapse {
                    Layout.fillWidth: true; title: tx("Параметры запуска")
                    Flow {
                        Layout.fillWidth: true; spacing: 5
                        Repeater { model: ((tv ? tv.chips : []) || []).length; delegate: Chip { required property int index; readonly property var modelData: ((tv ? tv.chips : []) || [])[index] || []; text: modelData[0] + " <b>" + modelData[1] + "</b>" } }
                        Btn { size: "sm"; kind: "tint"; text: tx("Все заданные параметры"); onClicked: openWin("plan") }
                    }
                }
            }
            GridLayout {
                id: plots
                Layout.fillWidth: true
                columns: ui.ccols > 0 ? ui.ccols : Math.max(1, Math.min(4, Math.floor((width + 12) / 430)))
                columnSpacing: th.gap; rowSpacing: th.gap
                Repeater { model: chartOrder; delegate: PlotTile { required property string modelData; required property int index; Layout.fillWidth: true; k: modelData; n: index + 1 } }
            }
            Pane {
                id: jpane
                Layout.fillWidth: true; padV: Math.round(12 * th.d); clickable: true; onClicked: openWin("log")
                RowLayout {
                    Layout.fillWidth: true; spacing: 14
                    Txt { text: tx("Журнал"); font.weight: th.wth; Layout.alignment: Qt.AlignTop }
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: 3
                        Repeater {
                            model: ((tv && tv.journal ? tv.journal : []) || []).length
                            delegate: RowLayout {
                                required property int index
                                readonly property var modelData: ((tv && tv.journal ? tv.journal : []) || [])[index] || []
                                Layout.fillWidth: true; spacing: 12
                                Small { text: modelData.time; Layout.preferredWidth: 62 }
                                Small { text: modelData.event; Layout.preferredWidth: 96; color: th.ink2 }
                                Txt { id: jtx; text: modelData.text; Layout.fillWidth: true; font.pixelSize: th.fs - 1; Tip { text: jtx.truncated ? jtx.text : "" } }
                            }
                        }
                        Small { visible: !(tv && tv.journal && tv.journal.length); text: tx("Событий пока нет") }
                    }
                    Icon { name: "expand"; color: jpane.hovered ? th.ink2 : th.ink4; Layout.preferredWidth: 14; Layout.preferredHeight: 14; Layout.alignment: Qt.AlignTop }
                }
            }
        }
    }
    function trainAct(id) {
        if (id === "stop") return askStop()
        if (id === "chat") { chatView.pick(tv.name); return go("chat") }
        if (id === "bench") { benchView.pickModel(tv.name); return go("bench") }
        if (id === "setup") return go("setup")
        if (id === "resume") return B.resume()
    }
    function askStop() {
        if (!tv) return
        if (!ui.confirmStop) return B.stop()
        ask({title: tx("Остановить") + " «" + tv.name + "»?", text: tv.stopNote || "", ok: tx("Остановить"), danger: true, liveRun: tv.name}, (r) => { if (r) B.stop() })
    }
    function openWin(k) { if (!tv || !tv.name) return; B.openWin(k); winBox.open = true }

    // ва1
    property string rq: ""
    property string rf: "all"
    property string rsort: "когда"
    property int rdir: -1
    property var rsel: ({})
    function runSelNames() { const L = []; for (const k in rsel) if (rsel[k]) L.push(k); return L }
    function runSelCount() { return runSelNames().length }
    function askDelete(names) {
        names = names.filter(n => n !== liveName)
        if (!names.length) return
        const sz = B.runsSize(names)
        ask({title: names.length === 1 ? tx("Удалить") + " «" + names[0] + "»?" : tx("Удалить запуски") + " (" + names.length + ")?",
             text: tx("Журнал, токены и обученные веса") + " · " + sz + ". " + tx("Восстановление невозможно.") + " " + tx("Исходная модель не затрагивается."), ok: tx("Удалить"), danger: true},
            (r) => { if (r) { B.deleteRuns(names); const s = Object.assign({}, rsel); names.forEach(n => delete s[n]); rsel = s } })
    }
    component SortHead: Item {
        id: shd
        property string k: ""
        property string label: ""
        property bool alignR: false
        implicitHeight: 30
        RowLayout {
            anchors.fill: parent; anchors.rightMargin: shd.alignR ? 10 : 0; spacing: 3; layoutDirection: shd.alignR ? Qt.RightToLeft : Qt.LeftToRight
            Small { text: shd.label; color: rsort === shd.k ? th.ink : th.ink3; font.weight: rsort === shd.k ? th.wth : th.wt; font.pixelSize: th.fs - 2 }
            Icon { visible: rsort === shd.k; name: rdir > 0 ? "up" : "down"; color: th.ink2; Layout.preferredWidth: 10; Layout.preferredHeight: 10 }
            Item { Layout.fillWidth: true }
        }
        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
            onClicked: { if (rsort === shd.k) rdir = -rdir; else { rsort = shd.k; rdir = (shd.k === "имя" || shd.k === "цель" || shd.k === "режим") ? 1 : -1 } } }
    }
    component Check: Rectangle {
        id: ck
        property bool checked: false
        signal toggled()
        implicitWidth: 16; implicitHeight: 16; radius: 4
        color: checked ? th.accent : "transparent"; border.width: checked ? 0 : 1.5; border.color: th.ink4
        Icon { anchors.centerIn: parent; visible: ck.checked; name: "check"; color: th.onAccent; width: 12; height: 12; sw: 2 }
        MouseArea { anchors.fill: parent; anchors.margins: -4; cursorShape: Qt.PointingHandCursor; onClicked: ck.toggled() }
    }

    component RunsView: Item {
        id: rvw
        readonly property var rv: { hh; return B.runs(rq, rf, rsort, rdir) }
        // вз1
        readonly property real ww: Math.max(0, thead.width - 16 - 32 - 96)
        readonly property var cws: {
            const k = th.fs / 13.5, base = [120, 80, 140, 116, 86, 68, 66, 70, 120].map(x => x * k), vis = base.map(() => true)
            const fixed = () => base.reduce((a, x, i) => a + (i && vis[i] ? x : 0), 0)
            for (const h of [2, 7, 6, 1]) if (fixed() + base[0] > ww) vis[h] = false
            const w = base.map((x, i) => vis[i] ? x : 0)
            let extra = ww - fixed() - base[0]
            if (extra > 0) { const sh = vis[2] ? [.45, .3, .25] : [.7, 0, .3]; w[0] += extra * sh[0]; w[2] += extra * sh[1]; w[8] += extra * sh[2] }
            else w[0] = Math.max(60, ww - fixed())
            return w
        }
        readonly property var vis: cws.map(x => x > 0)
        function colW(i) { return cws[i] }
        function focusSearch() { rsearch.input.forceActiveFocus(); rsearch.input.selectAll() }
        Scroll {
            anchors.fill: parent; maxW: ap.wide ? 0 : 1560
            Pane {
                Layout.fillWidth: true; padV: Math.round(14 * th.d)
                RowLayout {
                    id: rbar
                    Layout.fillWidth: true; spacing: 10
                    Field { id: rsearch; Layout.fillWidth: true; Layout.maximumWidth: 240; Layout.minimumWidth: 120; icon: "search"; ph: tx("Поиск"); onEdited: (t) => rq = t }
                    Seg { small: rbar.width < 760; options: [["all", tx("Все")], ["идёт", tx("Выполняются")], ["завершено", tx("Завершены")], ["остановлено", tx("Остановлены")], ["ошибка", tx("Ошибки")]]; value: rf; onPicked: (v) => rf = v }
                    Item { Layout.fillWidth: true }
                    Small { visible: rbar.width > 980; text: tx("Запусков") + " <b>" + rv.count + "</b>"; textFormat: Text.StyledText }
                    Small { visible: rbar.width > 880; text: tx("Занято") + " <b>" + rv.size + "</b>"; textFormat: Text.StyledText }
                    Small { visible: rbar.width > 1080 && rv.free !== ""; text: tx("Свободно") + " <b>" + rv.free + "</b>"; textFormat: Text.StyledText }
                }
                Item {
                    id: thead
                    Layout.fillWidth: true; implicitHeight: 32
                    Row {
                        id: hdr; anchors.fill: parent; anchors.leftMargin: 8; anchors.rightMargin: 8; spacing: 0
                        Item { width: 32; height: parent.height; Check { anchors.verticalCenter: parent.verticalCenter; checked: rv.rows.length > 0 && rv.rows.every(r => rsel[r.name]); onToggled: { const s = {}; if (!checked) rv.rows.forEach(r => s[r.name] = true); rsel = s } } }
                        SortHead { width: rvw.colW(0); height: parent.height; k: "имя"; label: tx("Имя") }
                        SortHead { width: rvw.colW(1); height: parent.height; k: "цель"; label: tx("Способ") }
                        SortHead { width: rvw.colW(2); height: parent.height; k: "режим"; label: tx("Режим"); visible: rvw.vis[2] }
                        SortHead { width: rvw.colW(3); height: parent.height; k: "состояние"; label: tx("Состояние") }
                        SortHead { width: rvw.colW(4); height: parent.height; k: "шаг"; label: tx("Шаг"); alignR: true }
                        SortHead { width: rvw.colW(5); height: parent.height; k: "лучшая"; label: tx("Лучший"); alignR: true }
                        SortHead { width: rvw.colW(6); height: parent.height; k: "прошло"; label: tx("Время"); alignR: true; visible: rvw.vis[6] }
                        SortHead { width: rvw.colW(7); height: parent.height; k: "размер"; label: tx("Размер"); alignR: true; visible: rvw.vis[7] }
                        Item { width: 16; height: 1 }
                        SortHead { width: rvw.colW(8); height: parent.height; k: "когда"; label: tx("Когда") }
                    }
                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep }
                }
                Column {
                    Layout.fillWidth: true
                    Repeater {
                        model: (rv.rows || []).length
                        delegate: Rectangle {
                            id: rrow
                            required property int index
                            readonly property var modelData: (rv.rows || [])[index] || []
                            width: parent.width; height: Math.round(38 * th.d)
                            radius: th.rc
                            color: rsel[modelData.name] ? th.wash : rma.containsMouse ? th.fill : "transparent"
                            MouseArea { id: rma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { B.openRun(rrow.modelData.name); go("train") } }
                            Row {
                                anchors.fill: parent; anchors.leftMargin: 8; anchors.rightMargin: 8
                                Item { width: 32; height: parent.height
                                    Check { anchors.verticalCenter: parent.verticalCenter; checked: !!rsel[rrow.modelData.name]
                                            onToggled: { const s = Object.assign({}, rsel); if (s[rrow.modelData.name]) delete s[rrow.modelData.name]; else s[rrow.modelData.name] = true; rsel = s } } }
                                Txt { rightPadding: 10; width: rvw.colW(0); height: parent.height; text: rrow.modelData.name; font.weight: th.wth; Tip { text: rrow.modelData.data } }
                                Txt { rightPadding: 10; width: rvw.colW(1); height: parent.height; text: rrow.modelData.goal; color: th.ink2 }
                                Txt { rightPadding: 10; width: rvw.colW(2); height: parent.height; text: rrow.modelData.mode; color: th.ink2; visible: rvw.vis[2] }
                                Item { width: rvw.colW(3); height: parent.height; Pill { anchors.verticalCenter: parent.verticalCenter; kind: rrow.modelData.pill; text: rrow.modelData.word } }
                                Txt { rightPadding: 10; width: rvw.colW(4); height: parent.height; text: rrow.modelData.step; horizontalAlignment: Text.AlignRight }
                                Txt { rightPadding: 10; width: rvw.colW(5); height: parent.height; text: rrow.modelData.best; horizontalAlignment: Text.AlignRight; font.weight: th.wth }
                                Txt { rightPadding: 10; width: rvw.colW(6); height: parent.height; text: rrow.modelData.time; horizontalAlignment: Text.AlignRight; color: th.ink3; visible: rvw.vis[6] }
                                Txt { rightPadding: 10; width: rvw.colW(7); height: parent.height; text: rrow.modelData.size; horizontalAlignment: Text.AlignRight; color: th.ink3; visible: rvw.vis[7] }
                                Item { width: 16; height: 1 }
                                Txt { rightPadding: 10; width: rvw.colW(8); height: parent.height; text: rrow.modelData.when; color: th.ink3; Tip { text: rrow.modelData.whenFull } }
                                Row {
                                    width: 96; height: parent.height; spacing: 2; layoutDirection: Qt.RightToLeft
                                    Btn { anchors.verticalCenter: parent.verticalCenter; size: "sm"; kind: "danger"; icon: "trash"; tip: tx("Удалить"); on: !rrow.modelData.live; onClicked: askDelete([rrow.modelData.name]) }
                                    Btn { anchors.verticalCenter: parent.verticalCenter; size: "sm"; kind: "quiet"; icon: "chat"; tip: tx("Разговор"); on: rrow.modelData.weights; onClicked: { chatView.pick(rrow.modelData.name); go("chat") } }
                                    Btn { anchors.verticalCenter: parent.verticalCenter; size: "sm"; kind: "quiet"; icon: "open"; tip: tx("Открыть"); onClicked: { B.openRun(rrow.modelData.name); go("train") } }
                                }
                            }
                        }
                    }
                }
                Small { visible: rv.rows.length === 0; text: rv.empty; Layout.alignment: Qt.AlignHCenter; Layout.topMargin: 18; Layout.bottomMargin: 18 }
            }
            Pane {
                visible: !!(cm && cm.names && cm.names.length)
                Layout.fillWidth: true
                RowLayout {
                    Layout.fillWidth: true
                    H2 { text: tx("Сравнение") }
                    Small { text: cm && cm.names ? cm.names.join(" · ") : ""; Layout.fillWidth: true }
                    Btn { size: "sm"; kind: "quiet"; text: tx("Закрыть"); onClicked: B.closeCompare() }
                }
                Flow {
                    Layout.fillWidth: true; spacing: 14
                    Repeater { model: cm && cm.names ? cm.names : []; delegate: Row { required property string modelData; required property int index; spacing: 6
                        Rectangle { width: 12; height: 3; radius: 1.5; color: th.sc(cm.colors[index]); anchors.verticalCenter: parent.verticalCenter }
                        Small { text: modelData; color: th.ink2 } } }
                }
                GridLayout {
                    Layout.fillWidth: true; columns: width > 760 ? 2 : 1; columnSpacing: th.gap; rowSpacing: th.gap
                    Repeater {
                        model: cm && cm.charts ? cm.charts : []
                        delegate: Rectangle {
                            required property var modelData
                            required property int index
                            Layout.fillWidth: true; implicitHeight: 248; radius: th.r2; color: th.fill
                            Small { x: 12; y: 10; text: modelData.title; color: th.ink2 }
                            Plot { x: 8; y: 32; width: parent.width - 16; height: parent.height - 42; key: "cmp:" + index }
                        }
                    }
                }
                Column {
                    Layout.fillWidth: true; Layout.topMargin: 4
                    Item {
                        width: parent.width; height: Math.round(32 * th.d)
                        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep }
                        Row {
                            anchors.fill: parent
                            readonly property real cwid: (width - 220) / Math.max(1, cm && cm.names ? cm.names.length : 1)
                            Item { width: 220; height: parent.height }
                            Repeater {
                                model: cm && cm.names ? cm.names : []
                                delegate: Item {
                                    required property string modelData
                                    required property int index
                                    width: parent.cwid; height: parent.height
                                    Row {
                                        anchors.verticalCenter: parent.verticalCenter; spacing: 7; width: parent.width - 10
                                        Rectangle { width: 12; height: 3; radius: 1.5; color: th.sc(cm.colors[index]); anchors.verticalCenter: parent.verticalCenter }
                                        Txt { text: modelData; font.weight: th.wth; width: Math.min(implicitWidth, parent.width - 19) }
                                    }
                                }
                            }
                        }
                    }
                    Repeater {
                        model: cm && cm.rows ? cm.rows : []
                        delegate: Rectangle {
                            required property var modelData
                            width: parent.width; height: Math.round(32 * th.d); color: "transparent"
                            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep }
                            Row {
                                anchors.fill: parent
                                readonly property real cwid: (width - 220) / Math.max(1, modelData.length - 1)
                                Small { width: 220; height: parent.height; text: modelData[0] }
                                Repeater { model: modelData.slice(1); delegate: Txt { required property string modelData; width: parent.cwid; height: parent.height; text: modelData; font.pixelSize: th.fs - .5 } }
                            }
                        }
                    }
                }
            }
        }
    }

    // ви1
    component Bubble: ColumnLayout {
        id: bub
        property string who: "m"           // ви2
        property string name: ""
        property string text: ""
        property bool typing: false
        property string meta: ""
        property bool err: false
        spacing: 4
        Small { visible: bub.who !== "me" && bub.name !== ""; text: bub.name; font.pixelSize: th.fs - 2.5; Layout.alignment: bub.who === "me" ? Qt.AlignRight : Qt.AlignLeft; Layout.leftMargin: 4 }
        Text { id: probe; visible: false; text: bub.text; font: btx.font; textFormat: Text.PlainText }     // ви3
        Rectangle {
            visible: bub.text !== "" || bub.typing
            Layout.alignment: bub.who === "me" ? Qt.AlignRight : Qt.AlignLeft
            Layout.maximumWidth: bub.width * (bub.who === "me" ? .78 : 1)
            implicitWidth: Math.max(46, Math.min(bub.width * (bub.who === "me" ? .78 : 1), Math.ceil(probe.implicitWidth) + 26))
            implicitHeight: btx.implicitHeight + 18
            radius: Math.round(th.r * .85)
            color: bub.who === "me" ? th.accent : bub.err ? th.badW : th.fill2
            TextEdit {
                id: btx; x: 12; y: 9; width: parent.width - 24
                text: bub.text === "" && bub.typing ? " " : bub.text
                readOnly: true; selectByMouse: true; wrapMode: TextEdit.Wrap; renderType: TextEdit.NativeRendering
                color: bub.who === "me" ? th.onAccent : th.ink; selectionColor: bub.who === "me" ? Qt.rgba(1, 1, 1, .3) : th.wash2
                font.family: th.ff; font.pixelSize: th.fs
            }
            Row {
                visible: bub.typing && bub.text === ""; spacing: 4; anchors.left: parent.left; anchors.leftMargin: btx.x; anchors.bottom: parent.bottom; anchors.bottomMargin: 13
                Repeater { model: 3; delegate: Rectangle { required property int index; width: 5; height: 5; radius: 2.5; color: th.ink3
                    SequentialAnimation on opacity { running: bub.typing && th.motion; loops: Animation.Infinite
                        PauseAnimation { duration: index * 160 } NumberAnimation { to: .25; duration: 380 } NumberAnimation { to: 1; duration: 380 } PauseAnimation { duration: (2 - index) * 160 } } } }
            }
        }
        Small { visible: bub.meta !== ""; text: bub.meta; font.pixelSize: th.fs - 2.5; Layout.alignment: bub.who === "me" ? Qt.AlignRight : Qt.AlignLeft; Layout.leftMargin: 4 }
    }

    component ChatView: Item {
        id: chv
        property string sel: "__"
        property string base: "__"                 // вк1
        readonly property string sk: sel + "|" + base
        property var tabs: []                      // вк2
        property var convs: ({})
        property var rids: ({})
        property int pending: 0
        property var models: { hh; return B.chatModels() }
        property bool abOn: false
        property bool optsOpen: false
        property string loadText: ""
        property string cid: ""                    // вк3
        property var cids: ({})                    // вк4
        property var chats: []                     // вк5
        property var attach: null                  // вк6
        property string attachErr: ""
        property real loadFrac: -1
        ListModel { id: conv }
        readonly property var rr: hh ? hh["разговор"] : null
        readonly property bool loadedAny: !!(rr && rr["имя"] !== null && rr["имя"] !== undefined)
        readonly property bool ready: !!(rr && rr["готов"] && loadedAny && (sel === "" || rr["имя"] === sel))
        readonly property bool loading: loadedAny && !(rr && rr["готов"])
        function defaultSel() { const L = models; const tvn = tv && tv.name ? tv.name : ""; for (let i = 0; i < L.length; i++) if (L[i].k && L[i].k === tvn) return tvn; return L.length > 1 ? L[1].k : "" }
        function enter() { if (base === "__") base = hh && hh["база"] ? hh["база"] : ""
            refreshChats(); if (sel === "__" && chats.length) openChat(chats[0].id)
            if (sel === "__" || !models.some(m => m.k === sel)) switchTo(defaultSel(), base); composer.edit.forceActiveFocus() }
        readonly property var bases: { hh; return B.chatBases() }
        function baseName(b) { for (let i = 0; i < bases.length; i++) if (bases[i][0] === b) return bases[i][1].split(" · ")[0]; return b ? b.replace(/\.tq$/, "") : "BF16" }
        function tabOpen(k, b) { if (!tabs.some(t => t.k === k && t.b === b)) tabs = tabs.concat([{k: k, b: b}]) }
        function tabClose(i) { const t = tabs[i]; const c = Object.assign({}, convs); delete c[t.k + "|" + t.b]; convs = c
            const n = tabs.slice(); n.splice(i, 1); tabs = n
            if (t.k === sel && t.b === base) { const o = n.length ? n[Math.max(0, i - 1)] : {k: sel, b: base}; if (n.length) switchTo(o.k, o.b); else conv.clear() } }
        function hist() { const h = []; let q = null
            for (let i = 0; i < conv.count; i++) { const o = conv.get(i); if (o.who === "me") q = o.full || o.text; else if (o.who === "m" && q !== null && !o.err && !o.typing) { h.push([q, o.text]); q = null } }
            return h }
        function pick(k) { switchTo(k, base === "__" ? "" : base) }
        function store() { const a = []; for (let i = 0; i < conv.count; i++) a.push(Object.assign({}, conv.get(i))); const c = Object.assign({}, convs); c[sk] = a; convs = c }
        function switchTo(k, b) {
            if (b === undefined) b = base === "__" ? "" : base
            if (k === sel && b === base) return
            if (sel !== "__") { store(); saveNow(); const ci = Object.assign({}, cids); ci[sk] = cid; cids = ci }
            sel = k; base = b; conv.clear(); cid = cids[sk] || ""
            const a = convs[sk] || []
            for (let i = 0; i < a.length; i++) conv.append(a[i])
            if (!k) abOn = false
            Qt.callLater(() => clist.positionViewAtEnd())
        }
        function blank(o) { return Object.assign({who: "m", name: "", text: "", full: "", typing: false, meta: "", err: false, aName: "", aText: "", aTyping: false, aMeta: "", aErr: false, bName: "", bText: "", bTyping: false, bMeta: "", bErr: false}, o) }
        function opts() { return {temp: +ui.chatTemp, max: +ui.chatMax, "сцена": ui.chatScene || "", "сила": +ui.chatScale, top_k: +ui.chatTopK, top_p: +ui.chatTopP,
                                  min_p: +ui.chatMinP, "повтор": +ui.chatRep, "зерно": +ui.chatSeed, ctx: +ui.chatCtx,
                                  "присутствие": +ui.chatPres, "частота": +ui.chatFreq, "окно": +ui.chatRepN, "думать": !!ui.chatThink, "база": base === "__" ? "" : base} }
        // вк7
        readonly property var sampPresets: ({"точно": {chatTemp: 0, chatTopK: 0, chatTopP: 1, chatMinP: 0, chatRep: 1},
                                             "ровно": {chatTemp: .5, chatTopK: 0, chatTopP: 1, chatMinP: 0, chatRep: 1},
                                             "живо": {chatTemp: .9, chatTopK: 0, chatTopP: .95, chatMinP: .05, chatRep: 1.1}})
        function authorsPreset() { const g = B.genDefaults(); const o = {chatTemp: .5, chatTopK: 0, chatTopP: 1, chatMinP: 0, chatRep: 1}
            if (g.temp !== undefined) o.chatTemp = g.temp; if (g.top_k !== undefined) o.chatTopK = g.top_k; if (g.top_p !== undefined) o.chatTopP = g.top_p
            if (g.min_p !== undefined) o.chatMinP = g.min_p; if (g["повтор"] !== undefined) o.chatRep = g["повтор"]; return o }
        function presetNow() { const ks = ["chatTemp", "chatTopK", "chatTopP", "chatMinP", "chatRep"]
            const same = (o) => ks.every(k => Math.abs(+ui[k] - +o[k]) < 1e-6)
            for (const n in sampPresets) if (same(sampPresets[n])) return n
            return same(authorsPreset()) ? "авторы" : "своё" }
        function send() {
            const q = composer.text.trim(); if (!q || pending) return
            composer.text = ""
            voiceCfg(); B.voiceStop()
            const k = sel, key = sk, h = hist()
            const full = attach ? "Файл «" + attach.name + "»" + (attach.cut ? " (обрезан)" : "") + ":\n```\n" + attach.text + "\n```\n\n" + q : q
            const shown = attach ? "[Файл " + attach.name + "] " + q : q
            attach = null; attachErr = ""
            tabOpen(sel, base === "__" ? "" : base)
            conv.append(blank({who: "me", text: shown, full: full}))
            const r = Object.assign({}, rids)
            if (abOn && k) {
                conv.append(blank({who: "ab", aName: k, bName: tx("Исходная"), aTyping: true, bTyping: true}))
                const idx = conv.count - 1
                r[B.ask(Object.assign(opts(), {"имя": k, base: false, q: full, new: true}))] = {key: key, idx: idx, part: "a"}
                r[B.ask(Object.assign(opts(), {"имя": k, base: true, q: full, new: true}))] = {key: key, idx: idx, part: "b"}
                pending += 2
            } else {
                conv.append(blank({who: "m", name: k || tx("Исходная"), typing: true}))
                r[B.ask(Object.assign(opts(), {"имя": k, base: !k, q: full, "история": h}))] = {key: key, idx: conv.count - 1, part: ""}
                pending += 1
            }
            rids = r
            clist.positionViewAtEnd()
        }
        function patch(m, f) {
            if (m.key === sk) { const o = conv.get(m.idx); if (!o) return; const p = f(o); for (const x in p) conv.setProperty(m.idx, x, p[x]) }
            else { const a = convs[m.key]; if (!a || !a[m.idx]) return; Object.assign(a[m.idx], f(a[m.idx])) }
        }
        function onTok(rid, piece) {
            const m = rids[rid]; if (!m) return
            if (m.part === "" && ui.voiceProv && ui.voiceMode !== "full") B.voiceFeed(rid, piece)
            const pre = m.part === "a" ? "aText" : m.part === "b" ? "bText" : "text"
            patch(m, (o) => { const p = {}; p[pre] = o[pre] + piece; return p })
            if (m.key === sk && clist.atYEnd) Qt.callLater(() => clist.positionViewAtEnd())
        }
        function onEnd(rid, info) {
            const m = rids[rid]; if (!m) return
            if (m.part === "" && ui.voiceProv) { if (ui.voiceMode === "full") { if (!info.error && !info.stopped) B.voiceSay(info.text) } else B.voiceEnd(rid) }
            const pre = m.part === "a" ? "a" : m.part === "b" ? "b" : ""
            const K = (x) => pre ? pre + x.charAt(0).toUpperCase() + x.slice(1) : x
            patch(m, (o) => { const p = {}; p[K("typing")] = false
                if (info.error) { p[K("text")] = (o[K("text")] || "") === "" ? info.error : o[K("text")]; p[K("meta")] = tx("Ошибка"); p[K("err")] = (o[K("text")] || "") === "" }
                else { p[K("meta")] = info.meta; p[K("text")] = info.stopped ? (o[K("text")] || "") : info.text }
                return p })
            const r = Object.assign({}, rids); delete r[rid]; rids = r
            pending = Math.max(0, pending - 1)
            B.refresh()
            saveKey(m.key)
        }
        // вк8
        function voiceCfg() { B.voiceConf(JSON.stringify({provider: ui.voiceProv || "", xtts_dir: ui.voiceXtts || "", speaker: ui.voiceSpk || "", lang: ui.voiceLang || "ru", device: ui.voiceDev || "auto"})) }
        function mkVoiceOpts() { voiceCfg(); const o = [["", tx("Выключено")]]; for (const p of B.voiceList()) o.push([p.id, p.name + (p.why ? " — " + p.why : "")]); return o }
        readonly property var voiceOpts: { ui.voiceXtts; ui.voiceProv; return mkVoiceOpts() }
        function newId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8) }
        function refreshChats() { chats = B.chatList() }
        function writeChat(id, key, msgs) {
            const i = key.indexOf("|"), first = msgs.find(m => m.who === "me")
            B.chatSave(id, JSON.stringify({id: id, title: first ? first.text.replace(/\s+/g, " ").trim().slice(0, 60) : "", model: key.slice(0, i),
                                           base: key.slice(i + 1), updated: Date.now(), messages: msgs}))
        }
        // вк9
        function saveKey(key) {
            const cur = key === sk
            if (cur ? (!conv.count || sel === "__") : !(convs[key] || []).length) return
            const msgs = []
            if (cur) for (let i = 0; i < conv.count; i++) msgs.push(Object.assign({}, conv.get(i)))
            else for (const m of convs[key]) msgs.push(Object.assign({}, m))
            for (const m of msgs) { m.typing = false; m.aTyping = false; m.bTyping = false }
            const id = cids[key] || newId()
            if (cur) cid = id
            const ci = Object.assign({}, cids); ci[key] = id; cids = ci
            writeChat(id, key, msgs); refreshChats()
        }
        function saveNow() { if (sel !== "__") saveKey(sk) }
        function newChat() { if (pending) return; saveNow(); cid = ""; conv.clear(); attach = null; attachErr = ""
            const c = Object.assign({}, convs); c[sk] = []; convs = c; const ci = Object.assign({}, cids); ci[sk] = ""; cids = ci
            composer.edit.forceActiveFocus() }
        function openChat(id) {
            if (pending) return
            const d = B.chatGet(id); if (!d || !d.messages) { refreshChats(); return }
            saveNow()
            sel = d.model && models.some(m => m.k === d.model) ? d.model : defaultSel()
            base = d.base && bases.some(b => b[0] === d.base) ? d.base : ""
            conv.clear(); const ms = d.messages.map(m => blank(m))
            for (const m of ms) conv.append(m)
            const c = Object.assign({}, convs); c[sk] = ms; convs = c
            cid = d.id; const ci = Object.assign({}, cids); ci[sk] = cid; cids = ci
            if (!sel) abOn = false
            refreshChats(); Qt.callLater(() => clist.positionViewAtEnd())
        }
        function deleteChat(id) { B.chatDrop(id)
            if (id === cid) { cid = ""; conv.clear(); const c = Object.assign({}, convs); c[sk] = []; convs = c }
            const ci = Object.assign({}, cids); for (const k in ci) if (ci[k] === id) ci[k] = ""; cids = ci
            refreshChats() }
        function attachFile() { const p = B.pick("файл"); if (!p) return
            const r = B.fileText(p, Math.max(400, Math.round(+ui.chatCtx * 2)))
            attach = r.error ? null : r; attachErr = r.error || "" }
        function fresh() { conv.clear(); const c = Object.assign({}, convs); c[sk] = []; convs = c; B.ask({"имя": sel, base: !sel, new: true, "база": base === "__" ? "" : base}) }
        Connections {
            target: B
            function onChatTok(rid, piece) { chv.onTok(rid, piece) }
            function onChatEnd(rid, info) { chv.onEnd(rid, info) }
            function onChatLoad(t, f) { chv.loadText = t; chv.loadFrac = f }
        }
        RowLayout {
            anchors.fill: parent; spacing: th.gap
            Pane {
                visible: !compactChat; Layout.preferredWidth: 236; Layout.fillHeight: true; padH: 10
                H2 { text: tx("Разговоры"); Layout.leftMargin: 10 }
                Btn { Layout.fillWidth: true; Layout.leftMargin: 4; Layout.rightMargin: 4; size: "sm"; kind: "quiet"; icon: "plus"; text: tx("Новый разговор"); onClicked: chv.newChat() }
                Scroll {
                    Layout.fillWidth: true; Layout.preferredHeight: Math.min(chv.height * 0.35, chv.chats.length * 48 + 6); padB: 4; spacing: 2
                    Repeater {
                        model: chv.chats.length
                        delegate: Rectangle {
                            required property int index
                            readonly property var cd: chv.chats[index] || {}
                            Layout.fillWidth: true; implicitHeight: ccl.implicitHeight + 12; radius: th.rc
                            color: cd.id === chv.cid ? th.wash : mch.containsMouse ? th.fill : "transparent"
                            ColumnLayout { id: ccl; x: 10; y: 6; width: parent.width - 44; spacing: 1
                                Txt { text: cd.title || tx("Без названия"); font.weight: th.wth; Layout.fillWidth: true; elide: Text.ElideRight; color: th.ink }
                                Small { text: (cd.model || tx("Исходная")) + " · " + cd.n; Layout.fillWidth: true; elide: Text.ElideRight; font.pixelSize: th.fs - 2.5 } }
                            MouseArea { id: mch; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: chv.openChat(cd.id) }
                            Btn { anchors.right: parent.right; anchors.rightMargin: 4; anchors.verticalCenter: parent.verticalCenter; size: "sm"; kind: "quiet"; icon: "trash"; tip: tx("Удалить разговор"); onClicked: chv.deleteChat(cd.id) }
                        }
                    }
                }
                H2 { text: tx("Модели"); Layout.leftMargin: 10 }
                Scroll {
                    Layout.fillWidth: true; Layout.fillHeight: true; Layout.preferredHeight: chv.height - 80; padB: 4; spacing: 2
                    Repeater {
                        model: (chv.models || []).length
                        delegate: Rectangle {
                            required property int index
                            readonly property var modelData: (chv.models || [])[index] || []
                            Layout.fillWidth: true; implicitHeight: mcl.implicitHeight + 14; radius: th.rc
                            color: modelData.k === chv.sel ? th.wash : mma2.containsMouse ? th.fill : "transparent"
                            ColumnLayout { id: mcl; x: 10; y: 7; width: parent.width - 20; spacing: 1
                                Txt { text: modelData.n; font.weight: th.wth; Layout.fillWidth: true; color: modelData.k === chv.sel ? th.accentInk : th.ink }
                                Small { text: modelData.s; Layout.fillWidth: true; font.pixelSize: th.fs - 2.5 } }
                            MouseArea { id: mma2; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: chv.switchTo(modelData.k, chv.base === "__" ? "" : chv.base) }
                        }
                    }
                }
            }
            Rectangle {
                id: ccard
                Layout.fillWidth: true; Layout.fillHeight: true
                radius: th.r; color: th.card; border.width: th.hair; border.color: th.edge
                readonly property bool roomy: width > 780
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: Math.round(18 * th.d); anchors.leftMargin: Math.round(20 * th.d); anchors.rightMargin: Math.round(20 * th.d); spacing: 10
                    RowLayout {
                        Layout.fillWidth: true; spacing: 10
                        H2 { visible: !compactChat; text: chv.sel ? chv.sel : tx("Исходная модель"); Layout.fillWidth: true; Layout.maximumWidth: implicitWidth; Layout.minimumWidth: Math.min(implicitWidth, 72) }
                        Combo { visible: compactChat; implicitWidth: 200; options: chv.models.map(m => [m.k, m.n]); value: chv.sel; onPicked: (v) => chv.switchTo(v, chv.base === "__" ? "" : chv.base) }
                        Pill { kind: chv.ready ? "ok" : chv.loading ? "live" : ""
                               text: chv.ready ? tx("Загружена") : chv.loading ? tx("Загружается") : chv.loadedAny ? tx("В памяти") + ": " + (chv.rr["имя"] || tx("Исходная")) : tx("Не загружена")
                               tip: chv.ready ? "" : tx("Веса загружаются при первом вопросе.") }
                        Btn { visible: chv.loadedAny; kind: "quiet"; size: "sm"; icon: "power"
                              tip: tx("Выгрузить модель") + "\n" + tx("Освобождает ОЗУ и видеопамять. Текущий ответ прерывается, при следующем вопросе модель загружается заново."); onClicked: B.chatStop() }
                        Item { Layout.fillWidth: true }
                        Row {
                            spacing: 8; opacity: chv.sel ? 1 : .42
                            Sw { checked: chv.abOn; on: chv.sel !== ""; onToggled: (v) => chv.abOn = v; anchors.verticalCenter: parent.verticalCenter }
                            Txt {
                                visible: ccard.roomy; text: tx("Сравнение с исходной"); color: th.ink2; anchors.verticalCenter: parent.verticalCenter
                                MouseArea { anchors.fill: parent; enabled: chv.sel !== ""; cursorShape: Qt.PointingHandCursor; onClicked: chv.abOn = !chv.abOn }
                            }
                            Tip { text: tx("Каждый вопрос направляется обученной и исходной моделям, диалог начинается с чистого листа.") }
                        }
                        Btn { kind: chv.optsOpen ? "tint" : "quiet"; size: "sm"; icon: "settings"; text: ccard.roomy ? tx("Параметры") : ""; tip: ccard.roomy ? "" : tx("Параметры"); onClicked: chv.optsOpen = !chv.optsOpen }
                        Btn { kind: "quiet"; size: "sm"; icon: "edit"; text: ccard.roomy ? tx("Новый разговор") : ""; tip: ccard.roomy ? "" : tx("Новый разговор"); onClicked: chv.fresh() }
                    }
                    Flow {   // вк10
                        Layout.fillWidth: true; spacing: 6
                        Small { text: tx("Веса"); height: 26; verticalAlignment: Text.AlignVCenter; Tip { text: tx("Формат весов для этой вкладки: BF16 или квант. Обученная модель применяется поверх той же базы. Если нужного кванта нет, выбор запускает квантование.") } }
                        Combo { implicitWidth: 170; height: 26; options: chv.bases; value: chv.base === "__" ? "" : chv.base
                                onPicked: (v) => { if (String(v).indexOf("make:") === 0) { B.setupRun("квант:" + String(v).slice(5)); toast(tx("Квантование"), tx("База появится в списке после завершения квантования."), "info") } else chv.switchTo(chv.sel, v) } }
                        Item { width: 6; height: 1 }
                        Repeater { model: chv.tabs.length > 1 ? chv.tabs.length : 0
                            delegate: Rectangle { required property int index; readonly property var t: chv.tabs[index] || {k: "", b: ""}
                                readonly property bool on: t.k === chv.sel && t.b === chv.base
                                implicitHeight: 26; implicitWidth: trow.implicitWidth + 16; radius: 13; color: on ? th.wash : tma.containsMouse ? th.fill2 : th.fill
                                Row { id: trow; anchors.centerIn: parent; spacing: 6
                                    Txt { text: (t.k || tx("Исходная")) + " · " + chv.baseName(t.b); color: on ? th.accentInk : th.ink2; font.pixelSize: th.fs - 2; anchors.verticalCenter: parent.verticalCenter }
                                    Txt { text: "×"; color: th.ink3; anchors.verticalCenter: parent.verticalCenter
                                          MouseArea { anchors.fill: parent; anchors.margins: -4; cursorShape: Qt.PointingHandCursor; onClicked: chv.tabClose(index) } } }
                                MouseArea { id: tma; anchors.fill: parent; anchors.rightMargin: 18; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: chv.switchTo(t.k, t.b) } } }
                    }
                    GridLayout {
                        visible: chv.optsOpen; Layout.fillWidth: true; columns: width > 820 ? 4 : 2; columnSpacing: 12; rowSpacing: 8
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Температура"); color: th.ink2; Tip { text: tx("При значении 0 всегда выбирается самый вероятный ответ.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatTemp); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatTemp: Math.max(0, Math.min(2, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Длина ответа, токенов"); color: th.ink2 }
                            Field { Layout.fillWidth: true; text: String(ui.chatMax); onDone: (t) => { const v = parseInt(t); if (!isNaN(v)) uiSet({chatMax: Math.max(1, Math.min(+ui.chatCtx || B.ctxMax(), v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Сила обучения"); color: th.ink2; Tip { text: tx("1 — результат обучения без изменений, 0 — исходная модель, 1,5 — усиленный эффект.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatScale); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatScale: Math.max(0, Math.min(3, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Присутствие"); color: th.ink2; Tip { text: tx("Вычитается из логита каждого токена, уже встречавшегося в окне повтора. 0 — выключено.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatPres); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatPres: Math.max(0, Math.min(2, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Частота"); color: th.ink2; Tip { text: tx("Вычитается за каждое повторение токена в окне: чем чаще встречается слово, тем ниже вероятность его повтора. 0 — выключено.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatFreq); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatFreq: Math.max(0, Math.min(2, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Окно повтора, токенов"); color: th.ink2; Tip { text: tx("Число последних токенов, учитываемых штрафом повтора, присутствием и частотой.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatRepN); onDone: (t) => { const v = parseInt(t); if (!isNaN(v)) uiSet({chatRepN: Math.max(1, Math.min(4096, v))}) } } }
                        RowLayout { Layout.fillWidth: true; spacing: 8
                            Sw { checked: !!ui.chatThink; onToggled: (v) => uiSet({chatThink: v}) }
                            Txt { text: tx("Размышления"); color: th.ink2; Tip { text: tx("Модели с режимом размышлений (Qwen 3 и др.) сначала формируют рассуждение, затем ответ. При выключении ответ выдаётся сразу.") } } }
                        ColumnLayout { Layout.columnSpan: parent.columns; Layout.fillWidth: true; spacing: 5
                            Small { text: tx("Озвучка ответов"); color: th.ink2; Tip { text: tx("Озвучивание ответа модели выбранным TTS: по ходу ответа (предложение за предложением) или целиком после его завершения. Размышления и код не озвучиваются.") } }
                            Combo { Layout.fillWidth: true; options: chv.voiceOpts; value: ui.voiceProv || ""; onPicked: (v) => { uiSet({voiceProv: v}); chv.voiceCfg() } }
                            Combo { visible: !!ui.voiceProv; Layout.fillWidth: true; options: [["stream", tx("По ходу ответа")], ["full", tx("После ответа")]]; value: ui.voiceMode; onPicked: (v) => uiSet({voiceMode: v}) }
                            RowLayout { visible: ui.voiceProv === "xtts"; Layout.fillWidth: true; spacing: 6
                                Field { Layout.fillWidth: true; ph: tx("Папка XTTS"); text: ui.voiceXtts; onDone: (t) => { uiSet({voiceXtts: t}); chv.voiceCfg() } }
                                Btn { size: "sm"; icon: "folder"; tip: tx("Выбрать папку"); onClicked: { const p = B.pick("папка"); if (p) { uiSet({voiceXtts: p}); chv.voiceCfg() } } } }
                            Field { visible: ui.voiceProv === "xtts"; Layout.fillWidth: true; ph: tx("Голос: имя или файл wav с образцом"); text: ui.voiceSpk; onDone: (t) => { uiSet({voiceSpk: t}); chv.voiceCfg() } }
                            Field { visible: !!ui.voiceProv; Layout.fillWidth: true; ph: tx("Язык озвучки (ru, en и др.)"); text: ui.voiceLang; onDone: (t) => { uiSet({voiceLang: t.trim() || "ru"}); chv.voiceCfg() } }
                            Btn { visible: !!ui.voiceProv; size: "sm"; kind: "quiet"; text: tx("Остановить озвучку"); onClicked: B.voiceStop() } }
                        ColumnLayout { Layout.columnSpan: parent.columns; Layout.fillWidth: true; spacing: 5
                            RowLayout { Small { text: tx("Системная инструкция"); color: th.ink2 } Item { Layout.fillWidth: true }
                                Small { text: (ui.chatScene || "").length + " " + tx("знаков") }
                                Btn { size: "sm"; kind: "quiet"; text: tx("Очистить"); on: !!ui.chatScene; onClicked: { sceneA.edit.text = ""; uiSet({chatScene: ""}) } } }
                            Area { id: sceneA; Layout.fillWidth: true; minH: 110; ph: tx("Роль модели, правила и формат ответа. При изменении инструкции разговор начинается заново."); text: ui.chatScene || ""
                                   onEdited: sceneT.restart() }
                            Timer { id: sceneT; interval: 700; onTriggered: uiSet({chatScene: sceneA.edit.text}) } }
                        RowLayout { Layout.columnSpan: parent.columns; Layout.fillWidth: true; spacing: 10
                            Small { text: tx("Выборка"); color: th.ink2 }
                            Seg { small: true; options: [["точно", tx("Точная")], ["ровно", tx("Ровная")], ["живо", tx("Вариативная")], ["авторы", tx("Рекомендация авторов")], ["своё", tx("Собственная")]]
                                  value: chv.presetNow(); onPicked: (v) => { if (v === "авторы") uiSet(chv.authorsPreset()); else if (chv.sampPresets[v]) uiSet(chv.sampPresets[v]) } }
                            Tip { text: tx("Точная — всегда самый вероятный токен. Ровная — температура 0,5 без отсечек. Вариативная — температура 0,9, top-p 0,95, min-p 0,05, штраф повтора 1,1. Рекомендация авторов — значения из generation_config.json модели.") }
                            Item { Layout.fillWidth: true } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: "Top-k"; color: th.ink2; Tip { text: tx("Число наиболее вероятных токенов, из которых делается выбор. 0 — без ограничения.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatTopK); onDone: (t) => { const v = parseInt(t); if (!isNaN(v)) uiSet({chatTopK: Math.max(0, Math.min(1000, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: "Top-p"; color: th.ink2; Tip { text: tx("Выбор из наиболее вероятных токенов, пока их суммарная вероятность не достигнет p. 1 — выключено.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatTopP); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatTopP: Math.max(.01, Math.min(1, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: "Min-p"; color: th.ink2; Tip { text: tx("Отбрасываются токены, вероятность которых ниже заданной доли от вероятности лучшего. 0 — выключено.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatMinP); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatMinP: Math.max(0, Math.min(1, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Штраф повтора"); color: th.ink2; Tip { text: tx("Делит вероятность токенов из последних 256. 1 — выключено, 1,1 — типичное значение.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatRep); onDone: (t) => { const v = parseFloat(t.replace(",", ".")); if (!isNaN(v)) uiSet({chatRep: Math.max(1, Math.min(2, v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Контекст, токенов"); color: th.ink2; Tip { text: tx("Суммарная длина вопроса, предыдущих реплик и ответа. При изменении длины модель перезагружается. Чем больше длина, тем больше расход видеопамяти.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatCtx); onDone: (t) => { const v = parseInt(t); if (!isNaN(v)) uiSet({chatCtx: Math.max(256, Math.min(B.ctxMax(), v))}) } } }
                        ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Зерно"); color: th.ink2; Tip { text: tx("При одном и том же значении на один и тот же вопрос выдаётся один и тот же ответ. −1 — новое значение при каждом запросе.") } }
                            Field { Layout.fillWidth: true; text: String(ui.chatSeed); onDone: (t) => { const v = parseInt(t.replace("−", "-")); if (!isNaN(v)) uiSet({chatSeed: Math.max(-1, v)}) } } }
                    }
                    ColumnLayout {
                        visible: chv.loadText !== ""; Layout.fillWidth: true; spacing: 6
                        Small { text: chv.loadText; color: th.ink2; Layout.fillWidth: true }
                        Bar { Layout.fillWidth: true; frac: chv.loadFrac }
                    }
                    Item {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        ColumnLayout {
                            visible: conv.count === 0; anchors.centerIn: parent; spacing: 8; width: parent.width
                            Icon { name: "chat"; color: th.ink4; Layout.preferredWidth: 32; Layout.preferredHeight: 32; Layout.alignment: Qt.AlignHCenter; sw: 1.2 }
                            Txt { text: chv.sel ? chv.sel : tx("Исходная модель"); font.pixelSize: th.fs + 3; font.weight: th.wth; Layout.alignment: Qt.AlignHCenter }
                            Small {
                                Layout.alignment: Qt.AlignHCenter
                                text: { for (let i = 0; i < chv.models.length; i++) { const m = chv.models[i]; if (m.k === chv.sel) return m.k ? [m.goal, m.best ? tx("Потеря (loss)") + " " + m.best : "", m.step ? tx("Шаг") + " " + m.step : "", m.when].filter(x => x).join(" · ") : [m.s, tx("Без обучения")].join(" · ") } return "" }
                            }
                        }
                        ListView {
                            id: clist; anchors.fill: parent; model: conv; spacing: 14; clip: true; boundsBehavior: Flickable.StopAtBounds
                            interactive: false
                            delegate: Item {
                                id: md
                                required property int index
                                required property string who
                                required property string name
                                required property string text
                                required property bool typing
                                required property string meta
                                required property bool err
                                required property string aName
                                required property string aText
                                required property bool aTyping
                                required property string aMeta
                                required property bool aErr
                                required property string bName
                                required property string bText
                                required property bool bTyping
                                required property string bMeta
                                required property bool bErr
                                width: clist.width - 12
                                height: md.who === "ab" ? abRow.implicitHeight : one.implicitHeight
                                Bubble { id: one; visible: md.who !== "ab"; width: parent.width; who: md.who === "me" ? "me" : "m"; name: md.name; text: md.text; typing: md.typing; meta: md.meta; err: md.err }
                                Row {
                                    id: abRow; visible: md.who === "ab"; width: parent.width; spacing: 12
                                    Bubble { width: (abRow.width - 12) / 2; who: "a"; name: md.aName; text: md.aText; typing: md.aTyping; meta: md.aMeta; err: md.aErr }
                                    Bubble { width: (abRow.width - 12) / 2; who: "b"; name: md.bName; text: md.bText; typing: md.bTyping; meta: md.bMeta; err: md.bErr }
                                }
                            }
                        }
                        WheelHandler { onWheel: (ev) => { clist.contentY = Math.max(clist.originY, Math.min(clist.originY + clist.contentHeight - clist.height, clist.contentY - (ev.pixelDelta.y !== 0 ? ev.pixelDelta.y : ev.angleDelta.y / 120 * 90))) } }
                    }
                    RowLayout {
                        visible: !!chv.attach || !!chv.attachErr; Layout.fillWidth: true; spacing: 8
                        Chip { visible: !!chv.attach; implicitHeight: 19; text: chv.attach ? tx("Файл") + ": " + chv.attach.name + (chv.attach.cut ? " · " + tx("Обрезан") : "") : "" }
                        Small { visible: !!chv.attachErr; text: chv.attachErr; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Btn { visible: !!chv.attach; size: "sm"; kind: "quiet"; text: tx("Убрать"); onClicked: chv.attach = null }
                    }
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Btn { kind: "quiet"; icon: "file"; tip: tx("Прикрепить текстовый файл"); onClicked: chv.attachFile(); Layout.alignment: Qt.AlignBottom }
                        Area {
                            id: composer; Layout.fillWidth: true; grow: true; minH: Math.round(36 + 2 * th.d); maxH: 200; ph: tx("Сообщение")
                            enterSends: true; onSubmit: chv.send()
                        }
                        Btn { kind: "primary"; icon: "send"; on: composer.text.trim() !== "" && chv.pending === 0; tip: tx("Отправить · Enter") + "\n" + tx("Новая строка · Shift+Enter"); onClicked: chv.send(); Layout.alignment: Qt.AlignBottom }
                    }
                }
            }
        }
    }
    readonly property bool compactChat: views.width < 820


    // вл1
    component ModelsView: Item {
        id: mv
        readonly property var hb: B.hub
        readonly property var sel: hb ? hb.sel : null
        property string openRepo: ""   // вл2
        function toggle(repo) { if (openRepo === repo) openRepo = ""; else { openRepo = repo; qf.text = repo; B.hubLook(repo, false) } }
        readonly property bool fetching: !!(hh && hh["настройка"] && hh["настройка_что"] === "веса")
        readonly property var cat: hb ? hb.cat : null
        property string sortKey: "downloads"
        property bool onlyOk: false
        readonly property var shownList: { const L = hb ? hb.list : []; return mv.onlyOk ? L.filter(r => r["поддержка"] === true) : L }
        function enter() { B.hubRefresh() }
        function srcName(s) { return s === "сеть" ? tx("Сайт Hugging Face") : s === "кэш" ? tx("Кэш") : s === "кэш (устаревший)" ? tx("Кэш (устаревший)") : tx(s || "—") }
        function catalog(fresh, limit) { B.hubCatalog(cf.text, mv.sortKey, limit || 30, !!fresh) }
        function dl(n) { return !n ? "0" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(n) }
        function look(q, fresh) { if (q) { openRepo = ""; qf.text = q; B.hubLook(q, !!fresh) } }
        function gb(n) { return n ? gibS(n) : "—" }
        function bn(n) { return !n ? "—" : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + "B" : (n / 1e6).toFixed(0) + "M" }
        function lastLog() { const L = ((B.job && B.job.log) || "").trim().split("\n"); return L.length ? L[L.length - 1] : "" }
        Scroll {
            id: msc
            anchors.fill: parent; maxW: ap.wide ? 0 : 1560
            Pane {
                Layout.fillWidth: true
                H2 { text: tx("Модель с Hugging Face") }
                Small { text: tx("Имя в формате «организация/модель» или ссылка. Сведения и прогноз формируются по заголовкам файлов до загрузки весов."); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                RowLayout {
                    Layout.fillWidth: true; spacing: 8
                    Field { id: qf; Layout.fillWidth: true; icon: "search"; ph: tx("Организация/модель") + " · https://huggingface.co/…"; onAccepted: mv.look(text, false) }
                    Btn { kind: "primary"; text: tx("Показать"); on: !(mv.hb && mv.hb.busy) && qf.text.trim() !== ""; onClicked: mv.look(qf.text, false) }
                }
                Small { visible: !!(mv.hb && (mv.hb.busy || mv.hb.err)); text: mv.hb && mv.hb.busy ? tx("Выполняется запрос") + ": " + mv.hb.busy : (mv.hb ? tx(mv.hb.err || "") : "")
                        color: mv.hb && mv.hb.err && !mv.hb.busy ? th.bad : th.ink3; Layout.fillWidth: true; wrapMode: Text.WordWrap }
            }
            Component {   // вл3
                id: cardComp
                Pane {
                    visible: !!(mv.sel && !mv.sel["ошибка"]); width: parent ? parent.width : implicitWidth
                    RowLayout {
                        Layout.fillWidth: true; spacing: 10
                        H2 { text: mv.sel ? mv.sel.repo : ""; Layout.maximumWidth: mv.width * .5 }
                        Pill { kind: mv.sel && mv.sel["поддержка"] ? (mv.sel["склейка"] ? "need" : "ok") : "bad"
                               text: !mv.sel ? "" : mv.sel["поддержка"] ? (mv.sel["склейка"] ? tx("Подходит, части объединяются после загрузки") : tx("Поддерживается движком")) : tx("Не поддерживается") }
                        Pill { kind: mv.sel && mv.sel["проверка"] === "карта" ? "ok" : ""
                               text: !mv.sel ? "" : mv.sel["проверка"] === "карта" ? tx("Проверено на видеокарте") : mv.sel["проверка"] === "формулы" ? tx("Формулы сверены с transformers") : tx("Не проверено")
                               tip: mv.sel && mv.sel["проверка"] === "формулы" ? tx("Прямой проход совпал с transformers на процессоре. На видеокарте эта модель ещё не запускалась.") : "" }
                        Item { Layout.fillWidth: true }
                        Btn { size: "sm"; kind: "quiet"; icon: "open"; text: tx("Страница модели"); onClicked: Qt.openUrlExternally(mv.sel["ссылка"]) }
                        Btn { size: "sm"; kind: "quiet"; icon: "build"; tip: tx("Повторить запрос"); on: !(mv.hb && mv.hb.busy); onClicked: mv.look(mv.sel.repo, true) }
                    }
                    Txt { visible: !!(mv.sel && mv.sel["почему"]); text: mv.sel ? tx(mv.sel["почему"] || "") : ""; color: th.bad; Layout.fillWidth: true; wrapMode: Text.WordWrap; elide: Text.ElideNone }
                    Flow {
                        Layout.fillWidth: true; spacing: 6
                        Repeater {
                            model: !mv.sel ? [] : [
                                "<b>" + mv.bn(mv.sel["параметров"]) + "</b> " + tx("параметров"),
                                "<b>" + (mv.sel["слоёв"] || "—") + "</b> " + tx("слоёв") + (mv.sel["полных"] !== undefined ? " · " + mv.sel["полных"] + " " + tx("полного внимания") : ""),
                                tx("Контекст") + " <b>" + (mv.sel["контекст"] || "—") + "</b>",
                                tx("Веса") + " <b>" + mv.gb(mv.sel["байт_весов"]) + "</b>" + (mv.sel["кусков"] > 1 ? " · " + mv.sel["кусков"] + " " + tx("частей") : "") + (mv.sel["основной"] ? " · " + mv.sel["основной"] : ""),
                                tx("Лицензия") + " <b>" + (mv.sel["лицензия"] || "—") + "</b>" + (mv.sel["закрыта"] ? " · " + tx("требуется согласие на сайте") : ""),
                                tx("Загрузок") + " <b>" + (mv.sel["загрузок"] || 0) + "</b>" + (mv.sel["изменена"] ? " · " + mv.sel["изменена"] : "")]
                            delegate: Chip { required property var modelData; text: modelData }
                        }
                    }
                    Small { visible: !!(mv.sel && mv.sel["сводка"]); text: mv.sel ? (mv.sel["сводка"] || "").split("\n")[0] : ""; Layout.fillWidth: true; wrapMode: Text.WordWrap; elide: Text.ElideNone }
                    Txt { visible: !!(mv.sel && mv.sel["прогноз"] && mv.sel["прогноз"].length); text: tx("Прогноз для этой машины"); font.weight: th.wth; Layout.topMargin: 6 }
                    Repeater {
                        model: mv.sel && mv.sel["прогноз"] ? mv.sel["прогноз"] : []
                        delegate: RowLayout {
                            required property var modelData
                            Layout.fillWidth: true; spacing: 10
                            Lamp { st: modelData["вид"] === "ok" ? "ok" : modelData["вид"] === "bad" ? "bad" : "warn" }
                            Txt { text: tx(modelData["что"]); color: th.ink2; Layout.preferredWidth: Math.round(mv.width * .2) }
                            Txt { text: tx(modelData["знач"]); Layout.preferredWidth: Math.round(mv.width * .3) }
                            Small { text: tx(modelData["как"]); Layout.fillWidth: true }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true; Layout.topMargin: 6; spacing: 10
                        Small { Layout.fillWidth: true; wrapMode: Text.WordWrap
                                text: mv.fetching ? mv.lastLog() : mv.sel ? tx("Папка") + ": " + (mv.sel["папка"] || "") : "" }
                        Btn { visible: !!(mv.sel && mv.sel["на_диске"]); kind: "primary"; text: tx("Сделать текущей"); onClicked: B.modelUse(mv.sel["папка"]) }
                        Btn { visible: !!(mv.sel && !mv.sel["на_диске"]); kind: "primary"; icon: "save"; text: mv.fetching ? tx("Выполняется загрузка") : tx("Скачать и сделать текущей")
                              on: !!(mv.sel && mv.sel["поддержка"]) && !mv.fetching && !liveName
                              tip: liveName ? tx("Выполняется обучение: модель будет заменена после его завершения.") : tx("Прежняя модель остаётся на диске и в списке ниже."); onClicked: B.hubGet(mv.sel.repo) }
                    }
                }
            }
            Loader { Layout.fillWidth: true; active: !mv.openRepo; visible: active && !!item && item.visible; sourceComponent: cardComp }
            Pane {
                Layout.fillWidth: true
                H2 { text: tx("На диске") }
                Repeater {
                    model: mv.hb ? mv.hb.local : []
                    delegate: RowLayout {
                        required property var modelData
                        Layout.fillWidth: true; spacing: 10
                        Lamp { st: modelData["текущая"] ? "ok" : modelData["годна"] ? "info" : modelData["склеить"] ? "need" : modelData["есть"] ? "bad" : "info" }
                        ColumnLayout { Layout.fillWidth: true; spacing: 2
                            Txt { text: (modelData["модель"] || "?") + (modelData.repo ? " · " + modelData.repo : ""); font.weight: modelData["текущая"] ? th.wth : th.wt; Layout.fillWidth: true }
                            Small { text: modelData["папка"] + " · " + mv.gb(modelData["байт"]) + (modelData["почему"] ? " · " + tx(modelData["почему"]) : ""); Layout.fillWidth: true } }
                        Pill { visible: !!modelData["текущая"]; kind: "ok"; text: tx("Текущая") }
                        Btn { visible: !modelData["текущая"] && !!modelData["есть"]; size: "sm"; text: tx("Сделать текущей"); on: !liveName; onClicked: B.modelUse(modelData["папка"]) }
                    }
                }
            }
            Pane {
                Layout.fillWidth: true
                RowLayout {
                    Layout.fillWidth: true; spacing: 10
                    H2 { text: tx("Каталог Hugging Face") }
                    Pill { kind: mv.cat && mv.cat.pend > 0 ? "wait" : ""; text: mv.cat && mv.cat.pend > 0 ? tx("Проверка поддержки") + ": " + mv.cat.pend : ""
                           tip: tx("Движок читает config.json и заголовки файлов каждой модели: несколько КиБ, без загрузки весов.") }
                    Item { Layout.fillWidth: true }
                    Btn { size: "sm"; kind: "quiet"; icon: "build"; tip: tx("Обновить список"); on: !(mv.cat && mv.cat.busy); onClicked: mv.catalog(true, mv.cat ? mv.cat.limit : 30) }
                }
                Small { text: tx("Текстовые модели с весами safetensors, получаемые непосредственно с сайта. Список обновляется раз в сутки; по выбранной модели отображаются сведения и прогноз."); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                RowLayout {
                    Layout.fillWidth: true; spacing: 8
                    Field { id: cf; Layout.fillWidth: true; icon: "search"; ph: tx("Поиск по названию или автору"); onAccepted: mv.catalog(false, 30) }
                    Combo { implicitWidth: 170; value: mv.sortKey
                            options: [["downloads", tx("По загрузкам")], ["likes", tx("По отметкам «нравится»")], ["trending", tx("Набирают популярность")], ["modified", tx("Недавно изменены")], ["created", tx("Новые")]]
                            onPicked: (v) => { mv.sortKey = v; mv.catalog(false, 30) } }
                    Btn { kind: "primary"; text: tx("Найти"); on: !(mv.cat && mv.cat.busy); onClicked: mv.catalog(false, 30) }
                }
                RowLayout {
                    Layout.fillWidth: true; spacing: 10
                    Tog { text: tx("Только поддерживаемые"); on: mv.onlyOk; tip: tx("Скрываются модели, которые движок не читает или которые ещё не проверены."); onToggled: mv.onlyOk = !mv.onlyOk }
                    Small { Layout.fillWidth: true; wrapMode: Text.WordWrap
                            color: mv.cat && mv.cat.ошибка && !mv.cat.busy ? th.warn : th.ink3
                            text: !mv.cat ? "" : mv.cat.busy ? tx("Выполняется запрос списка") : mv.cat.ошибка ? tx("Нет связи с Hugging Face. Источник") + ": " + mv.srcName(mv.cat.источник)
                                  : tx("Источник") + ": " + mv.srcName(mv.cat.источник) + " · " + mv.shownList.length + " " + tx("из") + " " + (mv.hb ? mv.hb.list.length : 0)
                            Tip { text: mv.cat && !mv.cat.busy ? (mv.cat.ошибка || "") : "" } }
                }
                Repeater {
                    model: mv.shownList
                    delegate: Rectangle {
                        id: crec
                        required property var modelData
                        readonly property bool open: mv.openRepo === modelData.repo
                        Layout.fillWidth: true; implicitHeight: ccol.implicitHeight + 12; radius: th.rc
                        color: cma.containsMouse || open ? th.fill : "transparent"
                        ColumnLayout {
                        id: ccol; x: 8; y: 6; width: parent.width - 16; spacing: 8
                        RowLayout {
                            id: crow; Layout.fillWidth: true; spacing: 10
                            Txt { text: modelData.repo; Layout.preferredWidth: Math.round(mv.width * .3); elide: Text.ElideMiddle }
                            Small { text: tx(modelData["заметка"] || ""); Layout.fillWidth: true; elide: Text.ElideRight }
                            Small { text: mv.bn(modelData["всего_параметров"]); Layout.preferredWidth: 44; horizontalAlignment: Text.AlignRight
                                    Tip { text: tx("Общее число параметров, включая модули зрения и звука, если они есть.") } }
                            Small { text: "↓ " + mv.dl(modelData["загрузок"]); Layout.preferredWidth: 62; horizontalAlignment: Text.AlignRight
                                    Tip { text: tx("Загрузок за месяц") + " · ♥ " + (modelData["нравится"] || 0) } }
                            Row {      // вл4
                                Layout.preferredWidth: Math.round(mv.width * .34); Layout.alignment: Qt.AlignVCenter; layoutDirection: Qt.RightToLeft; spacing: 6
                                Pill { kind: modelData["проверка"] === "карта" ? "ok" : ""; text: modelData["проверка"] === "карта" ? tx("Проверено на видеокарте") : modelData["проверка"] === "формулы" ? tx("Формулы") : ""
                                       tip: modelData["проверка"] === "формулы" ? tx("Прямой проход совпал с transformers на процессоре. На видеокарте эта модель ещё не запускалась.") : "" }
                                Pill { kind: modelData["поддержка"] === true ? (modelData["склейка"] ? "need" : "ok") : modelData["поддержка"] === false ? "bad" : modelData["проверка_ошибка"] ? "warn" : "wait"
                                       text: modelData["поддержка"] === true ? (modelData["склейка"] ? tx("Объединение") : tx("Поддерживается")) : modelData["поддержка"] === false ? tx("Нет") : modelData["проверка_ошибка"] ? "?" : tx("Проверка")
                                       tip: modelData["поддержка"] === false ? tx(modelData["почему"] || "") : modelData["проверка_ошибка"] ? tx(modelData["проверка_ошибка"]) : modelData["склейка"] ? tx("Подходит, части объединяются после загрузки") : "" }
                                Pill { visible: !!modelData["на_диске"]; kind: "ok"; text: tx("На диске") }
                            }
                        }
                        Small { visible: crec.open && !(mv.sel && mv.sel.repo === modelData.repo); Layout.fillWidth: true
                                text: mv.hb && mv.hb.busy ? tx("Выполняется запрос") : (mv.hb && mv.hb.err ? tx(mv.hb.err) : "") }
                        Loader { Layout.fillWidth: true; active: crec.open && !!(mv.sel && mv.sel.repo === modelData.repo); visible: active; sourceComponent: cardComp }
                        }
                        MouseArea { id: cma; x: ccol.x; y: ccol.y; width: ccol.width; height: crow.height; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: mv.toggle(modelData.repo) }
                    }
                }
                Small { visible: !!(mv.cat && !mv.cat.busy && mv.shownList.length === 0); Layout.fillWidth: true; wrapMode: Text.WordWrap
                        text: mv.onlyOk && mv.hb && mv.hb.list.length ? tx("Среди загруженных моделей поддерживаемых нет. Можно отключить фильтр или показать больше моделей.") : tx("Ничего не найдено") }
                Btn { visible: !!(mv.cat && mv.cat.ещё); text: tx("Показать ещё"); on: !(mv.cat && mv.cat.busy); onClicked: mv.catalog(false, (mv.cat.limit || 30) + 100) }
            }
        }
    }

    component BenchView: Item {
        id: bv
        property string model: "__"
        property string src: "run"
        property string text: ""
        property string file: ""
        property string count: ""
        property string info: ""
        property bool cmp: true
        readonly property var models: { hh; return B.chatModels() }
        function enter() { if (model === "__" || !models.some(m => m.k === model)) model = models.length > 1 ? models[1].k : ""; fixSrc(); recount() }
        function pickModel(k) { model = k; fixSrc(); recount() }
        function fixSrc() { if (!model && src === "run") src = "list"; if (!model) cmp = false }
        function recount() {
            const r = B.questions(src, model, file, text)
            count = r.ok && r.qs.length ? r.qs.length + " " + plural(r.qs.length, tx("вопрос"), tx("вопроса"), tx("вопросов")) : ""
            info = r.ok ? r.info : r.why
        }
        Timer { id: bcT; interval: 250; onTriggered: bv.recount() }
        Scroll {
            anchors.fill: parent; maxW: ap.wide ? 0 : 1560
            Pane {
                Layout.fillWidth: true
                H2 { text: tx("Проверка вопросами") }
                GridLayout {
                    Layout.fillWidth: true; columns: width > 900 ? 4 : 2; columnSpacing: 12; rowSpacing: 10
                    ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Модель"); color: th.ink2 }
                        Combo { Layout.fillWidth: true; options: bv.models.map(m => [m.k, m.n]); value: bv.model; onPicked: (v) => { bv.model = v; bv.fixSrc(); if (v) bv.cmp = true; bv.recount() } } }
                    ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Вопросы"); color: th.ink2 }
                        Combo { Layout.fillWidth: true; options: bv.model ? [["run", tx("Из данных запуска")], ["file", tx("Из файла")], ["list", tx("Собственный список")]] : [["file", tx("Из файла")], ["list", tx("Собственный список")]]
                                value: bv.src; onPicked: (v) => { bv.src = v; bv.recount() } } }
                    ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Температура"); color: th.ink2 } Field { id: btemp; Layout.fillWidth: true; text: "0" } }
                    ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Длина ответа"); color: th.ink2 } Field { id: bmax; Layout.fillWidth: true; text: "48" } }
                }
                ColumnLayout {
                    visible: bv.src === "file"; Layout.fillWidth: true; spacing: 5
                    Small { text: tx("Файл"); color: th.ink2 }
                    RowLayout { Layout.fillWidth: true; spacing: 8
                        Field { Layout.fillWidth: true; text: bv.file; ph: tx("Путь к .jsonl или .txt"); onDone: (t) => { bv.file = t; bv.recount() } }
                        Btn { icon: "folder"; text: tx("Выбрать"); onClicked: { const p = B.pick("файл"); if (p) { bv.file = p; bv.recount() } } } }
                }
                ColumnLayout {
                    visible: bv.src === "list"; Layout.fillWidth: true; spacing: 5
                    RowLayout { Small { text: tx("Вопросы"); color: th.ink2 } Small { text: "· " + tx("Один на строку"); Tip { text: tx("Один вопрос на строку. Ожидаемый ответ указывается после « || » и используется для поиска совпадения.") } } }
                    Area { Layout.fillWidth: true; minH: 130; ph: tx("Вопрос || Ожидаемый ответ"); text: bv.text; onEdited: { bv.text = edit.text; bcT.restart() } }
                }
                ColumnLayout { Layout.fillWidth: true; spacing: 5; Small { text: tx("Системная инструкция"); color: th.ink2 } Field { id: bscene; Layout.fillWidth: true; ph: tx("Текст системной инструкции") } }
                RowLayout {
                    Layout.fillWidth: true; Layout.topMargin: 4; spacing: 10
                    Sw { checked: bv.cmp && !!bv.model; on: !!bv.model; onToggled: (v) => bv.cmp = v }
                    Txt { text: tx("Сравнить с исходной моделью"); color: bv.model ? th.ink : th.ink3 }
                    Item { Layout.fillWidth: true }
                    Small { text: bv.count }
                    Btn {
                        kind: "primary"; icon: "play"; text: tx("Запустить проверку"); on: !(bn && bn.run)
                        onClicked: B.benchRun({src: bv.src, run: bv.model, file: bv.file, text: bv.text, cmp: bv.cmp && !!bv.model,
                                               temp: parseFloat(btemp.text.replace(",", ".")) || 0, max: parseInt(bmax.text) || 48, scene: bscene.text})
                    }
                }
            }
            Pane {
                visible: !(bn && bn.n); Layout.fillWidth: true
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 8; Layout.topMargin: 18; Layout.bottomMargin: 18
                    Icon { name: "bench"; color: th.ink4; Layout.preferredWidth: 32; Layout.preferredHeight: 32; Layout.alignment: Qt.AlignHCenter; sw: 1.2 }
                    Txt { text: tx("Нет результатов"); font.pixelSize: th.fs + 3; font.weight: th.wth; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
                    Small { text: bv.info; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
                }
            }
            Pane {
                visible: !!(bn && bn.n); Layout.fillWidth: true
                RowLayout {
                    Layout.fillWidth: true; spacing: 8
                    H2 { text: tx("Итог") }
                    Small { text: bn ? bn.prog : ""; Layout.fillWidth: true }
                    Btn { size: "sm"; kind: "quiet"; icon: "copy"; text: tx("Копировать"); on: !(bn && bn.run); onClicked: B.benchCopy() }
                    Btn { size: "sm"; kind: "quiet"; icon: "save"; text: tx("Сохранить таблицу"); on: !(bn && bn.run); onClicked: B.benchSave() }
                    Btn { visible: !!(bn && bn.run); size: "sm"; kind: "danger"; icon: "stop"; text: tx("Остановить"); onClicked: B.benchStop() }
                }
                Flow {
                    Layout.fillWidth: true; spacing: 8
                    Repeater { model: ((bn ? bn.tiles : []) || []).length; delegate: Tile { required property int index; readonly property var modelData: ((bn ? bn.tiles : []) || [])[index] || []; width: 200; label: modelData[0]; value: modelData[1]; unit: modelData[2]; note: modelData[3] } }
                }
                Bar { visible: !!(bn && bn.run); Layout.fillWidth: true; frac: bn ? bn.frac : 0 }
                Item {
                    Layout.fillWidth: true; implicitHeight: 30
                    Row {
                        id: bhdr; anchors.fill: parent
                        readonly property bool cmp: !!(bn && bn.cmp)
                        readonly property bool hasE: !!(bn && bn.hasE)
                        readonly property real ww: width - 36 - 150
                        readonly property real cq: ww * (cmp ? (hasE ? .22 : .28) : (hasE ? .3 : .4))
                        readonly property real ce: hasE ? ww * .14 : 0
                        readonly property real ca: (ww - cq - ce) / (cmp ? 2 : 1)
                        Small { width: 36; height: parent.height; text: "#" }
                        Small { width: bhdr.cq; height: parent.height; text: tx("Вопрос") }
                        Small { width: bhdr.ce; height: parent.height; text: tx("Ожидается"); visible: bhdr.hasE }
                        Small { width: bhdr.ca; height: parent.height; text: bn && bn.model ? tx("Обученная") : tx("Исходная") }
                        Small { width: bhdr.ca; height: parent.height; text: tx("Исходная"); visible: bhdr.cmp }
                        Small { width: 150; height: parent.height; text: tx("Итог") }
                    }
                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep }
                }
                Column {
                    Layout.fillWidth: true
                    Repeater {
                        model: ((bn ? bn.rows : []) || []).length
                        delegate: Rectangle {
                            id: brw
                            required property int index
                            readonly property var modelData: ((bn ? bn.rows : []) || [])[index] || []
                            width: parent.width; height: Math.max(36, brr.implicitHeight + 16)
                            color: modelData.cls === "hit" ? Qt.rgba(th.ok.r, th.ok.g, th.ok.b, .06) : modelData.cls === "miss" ? Qt.rgba(th.bad.r, th.bad.g, th.bad.b, .05) : "transparent"
                            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep }
                            Row {
                                id: brr; y: 8; width: parent.width
                                Small { width: 36; text: brw.modelData.i }
                                Txt { width: bhdr.cq; rightPadding: 10; text: brw.modelData.q; wrapMode: Text.Wrap; elide: Text.ElideNone; Tip { text: brw.modelData.scene ? tx("Системная инструкция:") + " " + brw.modelData.scene : "" } }
                                Txt { width: bhdr.ce; rightPadding: 10; visible: bhdr.hasE; text: brw.modelData.exp; color: th.ink3; wrapMode: Text.Wrap; elide: Text.ElideNone }
                                Txt { width: bhdr.ca; rightPadding: 10; text: brw.modelData.a === null || brw.modelData.a === undefined ? "…" : brw.modelData.a; color: brw.modelData.a === null || brw.modelData.a === undefined ? th.ink4 : th.ink; wrapMode: Text.Wrap; elide: Text.ElideNone }
                                Txt { width: bhdr.ca; rightPadding: 10; visible: bhdr.cmp; text: brw.modelData.b === null || brw.modelData.b === undefined ? "…" : brw.modelData.b; color: brw.modelData.b === null || brw.modelData.b === undefined ? th.ink4 : th.ink2; wrapMode: Text.Wrap; elide: Text.ElideNone }
                                Txt {
                                    width: 150; text: brw.modelData.res; wrapMode: Text.Wrap; elide: Text.ElideNone; font.pixelSize: th.fs - 1
                                    color: brw.modelData.cls === "hit" ? th.ok : brw.modelData.cls === "miss" ? th.bad : th.ink2
                                    MouseArea { anchors.fill: parent; enabled: brw.modelData.mark; cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor; onClicked: B.benchMark(brw.modelData.i - 1) }
                                    Tip { text: brw.modelData.mark ? tx("Щелчок мышью отмечает результат вручную.") : "" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    function plural(n, a, b, c) { n = Math.abs(Math.round(n)); const t = n % 10, h = n % 100; return t === 1 && h !== 11 ? a : t >= 2 && t <= 4 && (h < 10 || h >= 20) ? b : c }

    // вн1
    readonly property var secs: [["look", tx("Оформление"), "palette"], ["charts", tx("Графики"), "chart"], ["behave", tx("Поведение"), "settings"],
                                 ["hw", tx("Оборудование"), "chip"], ["paths", tx("Пути"), "folder"], ["keys", tx("Клавиши"), "keys"], ["about", tx("О программе"), "info"]]
    component SRow: Item {
        id: sr
        property string label: ""
        property string tip: ""
        property bool wide: false            // во1
        default property alias ctl: src.data
        Layout.fillWidth: true
        implicitHeight: Math.max(Math.round(38 * th.d), src.implicitHeight + 12)
        RowLayout {
            anchors.fill: parent; spacing: 12
            Row {
                Layout.fillWidth: !sr.wide; Layout.minimumWidth: 0
                Layout.preferredWidth: sr.wide ? Math.min(300, sr.width * .38) : slab.implicitWidth + (sr.tip !== "" ? 20 : 0)
                spacing: 6
                Txt { id: slab; text: sr.label; color: th.ink; width: Math.min(implicitWidth, parent.width - (sr.tip !== "" ? 20 : 0)); anchors.verticalCenter: parent.verticalCenter }
                Icon { visible: sr.tip !== ""; name: "info"; color: th.ink4; width: 14; height: 14; anchors.verticalCenter: parent.verticalCenter; Tip { text: sr.tip } }
            }
            RowLayout { id: src; Layout.fillWidth: sr.wide; spacing: 8 }
        }
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep; opacity: .7 }
    }
    component PresetCard: Rectangle {
        id: pc
        property string name: ""
        property var a: ({})
        property bool own: false
        property bool sel: false
        signal picked()
        signal removed()
        readonly property var aa: Object.assign({}, apDef, a)
        readonly property bool lt: aa.theme === "light"
        implicitHeight: 104; radius: th.r2
        color: sel ? th.wash : pcm.containsMouse ? th.fill2 : th.fill
        border.width: sel ? 1.5 : 0; border.color: th.accent
        Wall { x: 6; y: 6; width: parent.width - 12; height: 66; radius: Math.max(4, pc.aa.radius / 2.5); preset: pc.aa.bg; light: pc.lt; solid: pc.aa.solid || "#0e0f13" }
        Rectangle { x: 14; y: 14; width: (parent.width - 28) * .3; height: 50; radius: Math.round(pc.aa.radius / 3); color: pc.lt ? Qt.rgba(.965, .965, .976, .85) : Qt.rgba(.1, .1, .12, .8) }
        Rectangle { x: 14 + (parent.width - 28) * .3 + 6; y: 14; width: (parent.width - 28) * .7 - 6; height: 24; radius: Math.round(pc.aa.radius / 3)
                    color: pc.lt ? Qt.rgba(1, 1, 1, .88) : Qt.rgba(.16, .16, .18, .88); border.width: pc.aa.line; border.color: pc.lt ? Qt.rgba(0, 0, 0, .1) : Qt.rgba(1, 1, 1, .14) }
        Rectangle { x: parent.width - 14 - 26; y: 52; width: 26; height: 8; radius: 4; color: pc.aa.accent }
        Txt { x: 10; y: 78; width: parent.width - 20 - (pc.own ? 18 : 0); text: pc.name + (pc.own ? "  · " + tx("Собственный") : ""); font.pixelSize: th.fs - 1; font.weight: Math.min(th.wt + 100, 600) }
        MouseArea { id: pcm; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: pc.picked() }
        Btn { visible: pc.own; x: parent.width - width - 4; y: 74; size: "sm"; kind: "quiet"; icon: "x"; tip: tx("Удалить набор"); implicitHeight: 22; onClicked: pc.removed() }
    }
    function applyPreset(p) { apSet(Object.assign({}, apDef, {side: ap.side, sidew: ap.sidew, wide: ap.wide, full: ap.full, motion: ap.motion, lw: ap.lw, cfill: ap.cfill, cgrid: ap.cgrid, cdots: ap.cdots}, p)) }
    function presetOn(p) { const a = Object.assign({}, apDef, p); return ["theme", "bg", "accent", "radius", "line", "glass", "ff", "fs", "weight", "density"].every(k => String(a[k]) === String(ap[k])) }

    component SettingsView: Item {
        id: stv
        property var hw: ({})
        property var paths: []
        property string confPath: ""
        function loadCfg() { const c = B.settings(); hw = c["железо"]; paths = c["пути"]; confPath = c["conf"]; pModel.text = c["свои"]["MODEL"]; pWork.text = c["свои"]["WORK"]; pRepo.text = c["свои"]["HF_REPO"]; pTok.text = c["свои"]["HF_TOKEN"] || ""
                             hwHead.text = hw["headroom_mib"] || ""; hwT1.text = hw["tier1_params"] || ""; hwCap.text = hw["mem_cap_pct"] || "98" }
        onVisibleChanged: if (visible) loadCfg()
        RowLayout {
            anchors.fill: parent; spacing: th.gap
            Rectangle {
                Layout.preferredWidth: win.compact ? 56 : 210; Layout.alignment: Qt.AlignTop; implicitHeight: navc.implicitHeight + 16
                radius: th.r; color: th.card; border.width: th.hair; border.color: th.edge
                ColumnLayout {
                    id: navc; x: 8; y: 8; width: parent.width - 16; spacing: 2
                    Repeater {
                        model: secs
                        delegate: Rectangle {
                            required property var modelData
                            Layout.fillWidth: true; implicitHeight: Math.round(30 * th.d + 2); radius: th.rc
                            color: sec === modelData[0] ? th.fill2 : snm.containsMouse ? th.fill : "transparent"
                            RowLayout { anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10; spacing: 10
                                Icon { name: modelData[2]; color: sec === modelData[0] ? th.accent : th.ink3; Layout.preferredWidth: 16; Layout.preferredHeight: 16 }
                                Txt { visible: !win.compact; text: modelData[1]; Layout.fillWidth: true; color: sec === modelData[0] ? th.ink : th.ink2; font.weight: Math.min(th.wt + 60, 600) } }
                            MouseArea { id: snm; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { sec = modelData[0]; ss.toTop() } }
                            Tip { text: win.compact ? modelData[1] : "" }
                        }
                    }
                }
            }
            Scroll {
                id: ss
                Layout.fillWidth: true; Layout.fillHeight: true; maxW: 980

                // вт1
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    RowLayout { Layout.fillWidth: true
                        H2 { text: tx("Наборы"); Layout.fillWidth: true }
                        Btn { size: "sm"; icon: "plus"; text: tx("Сохранить текущее оформление")
                              onClicked: ask({title: tx("Сохранить набор"), input: {ph: tx("Название")}, ok: tx("Сохранить")}, (n) => { if (!n) return
                                  const a = {}; ["theme", "bg", "solid", "accent", "radius", "line", "glass", "ff", "fs", "weight", "density"].forEach(k => a[k] = ap[k])
                                  uiSet({presets: (ui.presets || []).concat([{n: String(n).trim().slice(0, 40), a: a}])}); toast(tx("Набор сохранён"), n, "ok") }) }
                        Btn { size: "sm"; kind: "quiet"; icon: "copy"; text: tx("Экспорт"); onClicked: { B.copy(JSON.stringify(ap)); toast(tx("Оформление скопировано"), "", "ok") } }
                        Btn { size: "sm"; kind: "quiet"; text: tx("Импорт")
                              onClicked: ask({title: tx("Импорт оформления"), text: tx("Требуется вставить JSON, скопированный кнопкой «Экспорт»."), input: {area: true, ph: "{\"theme\": \"dark\", …}"}, ok: tx("Применить")},
                                  (t) => { if (!t) return; try { const a = JSON.parse(t); if (typeof a !== "object" || !a) throw 0; apSet(Object.assign({}, apDef, a)); toast(tx("Оформление применено"), "", "ok") } catch (e) { toast(tx("Некорректный JSON"), "", "bad") } }) }
                    }
                    GridLayout {
                        Layout.fillWidth: true; columns: Math.max(2, Math.floor(width / 168)); columnSpacing: 10; rowSpacing: 10
                        Repeater { model: presets; delegate: PresetCard { required property var modelData; Layout.fillWidth: true; name: modelData[0]; a: modelData[1]; sel: presetOn(modelData[1]); onPicked: applyPreset(modelData[1]) } }
                        Repeater { model: ui.presets || []; delegate: PresetCard { required property var modelData; required property int index; Layout.fillWidth: true; own: true; name: modelData.n; a: modelData.a; sel: presetOn(modelData.a)
                                   onPicked: applyPreset(modelData.a); onRemoved: { const p = (ui.presets || []).slice(); p.splice(index, 1); uiSet({presets: p}) } } }
                    }
                }
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    H2 { text: tx("Цвет") }
                    SRow { label: tx("Тема"); Seg { options: [["dark", tx("Тёмная")], ["light", tx("Светлая")]]; value: ap.theme; onPicked: (v) => apSet({theme: v}) } }
                    SRow {
                        label: tx("Акцент")
                        Flow {
                            Layout.fillWidth: true; spacing: 8
                            Repeater {
                                model: ["#0a84ff", "#5e5ce6", "#bf5af2", "#ff375f", "#c06c84", "#ff9f0a", "#ffd60a", "#30d158", "#64d2ff", "#8e8e93", "#e5e5ea", "#1d1d1f"]
                                delegate: Rectangle {
                                    required property string modelData
                                    width: 24; height: 24; radius: 12; color: modelData
                                    border.width: String(ap.accent).toLowerCase() === modelData ? 2 : 1; border.color: String(ap.accent).toLowerCase() === modelData ? th.ink : th.edge2
                                    Rectangle { anchors.centerIn: parent; visible: String(ap.accent).toLowerCase() === modelData; width: 8; height: 8; radius: 4; color: th.onAccent }
                                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: apSet({accent: modelData}) }
                                    Tip { text: modelData }
                                }
                            }
                            Btn { size: "sm"; kind: "quiet"; icon: "palette"; text: tx("Собственный цвет"); onClicked: { const c = B.pickColor(ap.accent); if (c) apSet({accent: c}) } }
                        }
                    }
                }
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    H2 { text: tx("Фон") }
                    GridLayout {
                        Layout.fillWidth: true; columns: Math.max(3, Math.floor(width / 112)); columnSpacing: 8; rowSpacing: 8
                        Repeater {
                            model: bgs.concat([["solid", tx("Собственный цвет"), ap.solid, ap.solid], ["image", tx("Изображение"), "", ""]])
                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true; implicitHeight: 78; radius: th.r2
                                color: ap.bg === modelData[0] ? th.wash : bgm.containsMouse ? th.fill2 : th.fill
                                border.width: ap.bg === modelData[0] ? 1.5 : 0; border.color: th.accent
                                Wall { x: 6; y: 6; width: parent.width - 12; height: 44; radius: Math.max(4, th.r2 - 4); visible: modelData[0] !== "image" || bgPath !== ""
                                       preset: modelData[0]; light: th.light; solid: ap.solid || "#0e0f13"; img: modelData[0] === "image" ? bgPath : ""; dim: ap.dim; blur: ap.iblur }
                                Rectangle { x: 6; y: 6; width: parent.width - 12; height: 44; radius: Math.max(4, th.r2 - 4); visible: modelData[0] === "image" && bgPath === ""; color: th.fill2
                                            Icon { anchors.centerIn: parent; name: "image"; color: th.ink3; width: 18; height: 18 } }
                                Txt { x: 9; y: 54; width: parent.width - 18; text: modelData[1]; font.pixelSize: th.fs - 1.5 }
                                MouseArea { id: bgm; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: { if (modelData[0] === "image" && !bgPath) return pickBg(); apSet({bg: modelData[0]}) } }
                            }
                        }
                    }
                    SRow { visible: ap.bg === "solid"; label: tx("Собственный цвет")
                        Rectangle { width: 28; height: 28; radius: 14; color: ap.solid; border.width: 1; border.color: th.edge2; MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: { const c = B.pickColor(ap.solid); if (c) apSet({solid: c, bg: "solid"}) } } }
                        Txt { text: ap.solid; color: th.ink3; font.family: "monospace"; font.pixelSize: th.fs - 1.5 } }
                    SRow { visible: ap.bg === "image"; label: tx("Изображение"); tip: "PNG, JPG, WebP, AVIF, GIF"
                        Btn { size: "sm"; icon: "image"; text: tx("Другой файл"); onClicked: pickBg() }
                        Btn { size: "sm"; kind: "quiet"; text: tx("Убрать"); on: bgPath !== ""; onClicked: { B.saveSettings({BG: ""}); bgPath = ""; apSet({bg: apDef.bg}) } } }
                    SRow { visible: ap.bg === "image"; label: tx("Затемнение изображения"); Sld { Layout.fillWidth: true; from: 0; to: .85; step: .05; value: ap.dim; out: Math.round(ap.dim * 100) + " %"; onMoved: (v) => apSet({dim: v}) } }
                    SRow { visible: ap.bg === "image"; label: tx("Размытие изображения"); Sld { Layout.fillWidth: true; from: 0; to: 40; step: 1; value: ap.iblur; out: ap.iblur + " px"; onMoved: (v) => apSet({iblur: v}) } }
                }
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    H2 { text: tx("Поверхности") }
                    SRow { label: tx("Непрозрачность"); tip: tx("Карточки и панели"); Sld { Layout.fillWidth: true; from: 0; to: 1; step: .02; value: ap.glass; out: Math.round(ap.glass * 100) + " %"; onMoved: (v) => apSet({glass: v}) } }
                    SRow { label: tx("Линии"); Seg { options: [[0, tx("Нет")], [.5, tx("Тонкие")], [1, tx("Обычные")], [1.5, tx("Чёткие")]]; value: ap.line; onPicked: (v) => apSet({line: +v}) } }
                    SRow { label: tx("Скругление"); Sld { Layout.fillWidth: true; from: 0; to: 28; step: 1; value: ap.radius; out: ap.radius + " px"; onMoved: (v) => apSet({radius: v}) } }
                }
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    H2 { text: tx("Текст") }
                    SRow { label: tx("Шрифт")
                        Combo { Layout.preferredWidth: 260; value: ap.ff; onPicked: (v) => apSet({ff: v})
                                options: [["inter", "Inter"], ["golos", "Golos Text"], ["onest", "Onest"], ["manrope", "Manrope"], ["jost", "Jost"], ["geologica", "Geologica"], ["system", tx("Системный")], ["mono", tx("Моноширинный")]] } }
                    SRow { label: tx("Размер"); Sld { Layout.fillWidth: true; from: 12; to: 17; step: .5; value: ap.fs; out: String(ap.fs).replace(".", ui.dec === "comma" ? "," : ".") + " px"; onMoved: (v) => apSet({fs: v}) } }
                    SRow { label: tx("Начертание"); Seg { options: [[400, tx("Обычное")], [500, tx("Плотное")]]; value: ap.weight; onPicked: (v) => apSet({weight: +v}) } }
                }
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    H2 { text: tx("Компоновка") }
                    SRow { label: tx("Плотность"); Seg { options: [[.85, tx("Плотно")], [1, tx("Обычно")], [1.15, tx("Просторно")]]; value: ap.density; onPicked: (v) => apSet({density: +v}) } }
                    SRow { label: tx("Панель"); Seg { options: [["left", tx("Слева")], ["right", tx("Справа")]]; value: ap.side; onPicked: (v) => apSet({side: v}) } }
                    SRow { label: tx("Ширина панели"); Sld { Layout.fillWidth: true; from: 200; to: 340; step: 4; value: ap.sidew; out: ap.sidew + " px"; onMoved: (v) => apSet({sidew: v}) } }
                    SRow { label: tx("Ширина содержимого"); Seg { options: [[0, tx("Ограничена")], [1, tx("Во всю ширину")]]; value: ap.wide; onPicked: (v) => apSet({wide: +v}) } }
                    SRow { label: tx("Во весь экран"); tip: tx("Вид окна, развёрнутого на весь экран."); Seg { options: [["gap", tx("С зазором")], ["edge", tx("Вплотную")]]; value: ap.full; onPicked: (v) => apSet({full: v}) } }
                    SRow { label: tx("Анимации"); Sw { checked: !!ap.motion; onToggled: (v) => apSet({motion: v ? 1 : 0}) } }
                }
                Pane {
                    visible: sec === "look"; Layout.fillWidth: true
                    H2 { text: tx("Сброс оформления") }
                    SRow { label: tx("Вернуть оформление по умолчанию")
                        Btn { kind: "danger"; text: tx("Сбросить"); onClicked: ask({title: tx("Сбросить оформление?"), text: tx("Собственные наборы сохраняются."), ok: tx("Сбросить"), danger: true}, (r) => { if (r) apSet(Object.assign({}, apDef)) }) } }
                }

                // вт2
                Pane {
                    visible: sec === "charts"; Layout.fillWidth: true
                    H2 { text: tx("Плитки на «Ходе обучения»") }
                    Repeater {
                        model: { const on = chartOrder; return on.concat(["loss", "acc", "vram", "temp", "time", "grad"].filter(k => on.indexOf(k) < 0)) }
                        delegate: SRow {
                            id: crow
                            required property string modelData
                            required property int index
                            readonly property bool on: chartOrder.indexOf(modelData) >= 0
                            label: ({loss: tx("Потеря (loss)"), acc: tx("Точность на ответе"), vram: tx("Видеопамять"), temp: tx("Температура"), time: tx("Длительность шага"), grad: tx("Норма градиента")})[modelData]
                            Sw { checked: crow.on; onToggled: (v) => { let a = chartOrder.slice(); if (v) a.push(crow.modelData); else a = a.filter(x => x !== crow.modelData); if (!a.length) a = ["loss"]; uiSet({charts: a}) } }
                            Item { Layout.fillWidth: true }
                            Btn { size: "sm"; kind: "quiet"; icon: "up"; tip: tx("Выше"); on: crow.on && chartOrder.indexOf(crow.modelData) > 0
                                  onClicked: { const a = chartOrder.slice(), i = a.indexOf(crow.modelData); a.splice(i, 1); a.splice(i - 1, 0, crow.modelData); uiSet({charts: a}) } }
                            Btn { size: "sm"; kind: "quiet"; icon: "down"; tip: tx("Ниже"); on: crow.on && chartOrder.indexOf(crow.modelData) < chartOrder.length - 1
                                  onClicked: { const a = chartOrder.slice(), i = a.indexOf(crow.modelData); a.splice(i, 1); a.splice(i + 1, 0, crow.modelData); uiSet({charts: a}) } }
                        }
                    }
                }
                Pane {
                    visible: sec === "charts"; Layout.fillWidth: true
                    H2 { text: tx("Вид графиков") }
                    SRow { label: tx("Колонок"); Seg { options: [[0, tx("Авто")], [1, "1"], [2, "2"], [3, "3"], [4, "4"]]; value: ui.ccols || 0; onPicked: (v) => uiSet({ccols: +v}) } }
                    SRow { label: tx("Высота графика"); Sld { Layout.fillWidth: true; from: 100; to: 280; step: 10; value: ui.ch || 150; out: (ui.ch || 150) + " px"; onMoved: (v) => uiSet({ch: v}) } }
                    SRow { label: tx("Толщина линии"); Sld { Layout.fillWidth: true; from: 1; to: 3.5; step: .25; value: ap.lw; out: String(ap.lw) + " px"; onMoved: (v) => apSet({lw: v}) } }
                    SRow { label: tx("Заливка под кривой"); Sw { checked: !!ap.cfill; onToggled: (v) => apSet({cfill: v ? 1 : 0}) } }
                    SRow { label: tx("Сетка"); Sw { checked: !!ap.cgrid; onToggled: (v) => apSet({cgrid: v ? 1 : 0}) } }
                    SRow { label: tx("Точки валидации"); Sw { checked: !!ap.cdots; onToggled: (v) => apSet({cdots: v ? 1 : 0}) } }
                }

                // вт3
                Pane {
                    visible: sec === "behave"; Layout.fillWidth: true
                    H2 { text: tx("Поведение") }
                    SRow { label: tx("Стартовый экран"); tip: tx("Режим «Авто» открывает ход обучения, если обучение выполняется.")
                        Seg { options: [["auto", tx("Авто")], ["setup", tx("Новое обучение")], ["train", tx("Ход обучения")], ["runs", tx("Запуски")]]; value: ui.start; onPicked: (v) => uiSet({start: v}) } }
                    SRow { visible: !!(hh && hh["уведомления"]); label: tx("Уведомление по окончании"); tip: tx("Уведомление на рабочем столе, в том числе при закрытом окне.")
                        Sw { checked: !!ui.notify; onToggled: (v) => uiSet({notify: v ? 1 : 0}) } }
                    SRow { visible: !!(hh && hh["трей"]); label: tx("В трей при закрытии"); tip: tx("Действует, пока выполняется обучение.")
                        Sw { checked: !!ui.tray; onToggled: (v) => uiSet({tray: v ? 1 : 0}) } }
                    SRow { label: tx("Подтверждать остановку"); Sw { checked: !!ui.confirmStop; onToggled: (v) => uiSet({confirmStop: v ? 1 : 0}) } }
                    SRow { label: tx("Десятичный знак"); Seg { options: [["dot", "0.5137"], ["comma", "0,5137"]]; value: ui.dec; onPicked: (v) => uiSet({dec: v}) } }
                }

                // вт4
                Pane {
                    visible: sec === "hw"; Layout.fillWidth: true
                    H2 { text: tx("Машина") }
                    GridLayout {
                        Layout.fillWidth: true; columns: Math.max(2, Math.min(5, Math.floor(width / 160))); columnSpacing: 8; rowSpacing: 8
                        readonly property var g: hh ? hh["гпу"] || {} : {}
                        readonly property var o: hh ? hh["озу"] || {} : {}
                        readonly property var v: hh ? hh["видеопамять"] || {} : {}
                        readonly property var dk: hh ? hh["диск"] || {} : {}
                        Tile { Layout.fillWidth: true; label: tx("Карта"); value: parent.g["название"] || parent.g["архитектура"] || "—"; note: parent.g["архитектура"] || ""; big: th.fs + 1 }
                        Tile { Layout.fillWidth: true; label: tx("Видеопамять"); value: parent.v["всего"] ? gibS(parent.v["всего"]) : parent.g["видеопамять"] ? gibS(parent.g["видеопамять"] * 1048576) : "—"; note: parent.v["всего"] ? tx("Занято") + " " + gibS(parent.v["занято"]) : "" }
                        Tile { Layout.fillWidth: true; label: tx("ОЗУ"); value: gibS(parent.o["всего"]); note: tx("Доступно") + " " + gibS(parent.o["доступно"]) }
                        Tile { Layout.fillWidth: true; label: tx("Шина"); value: parent.g["шина"] && parent.g["шина"]["поколение"] ? "PCIe " + parent.g["шина"]["поколение"] + ".0" : "—"; note: parent.g["шина"] && parent.g["шина"]["линий"] ? "×" + parent.g["шина"]["линий"] : "" }
                        Tile { Layout.fillWidth: true; label: tx("Диск"); value: gibS(parent.dk["свободно"]); note: tx("Свободно из") + " " + gibS(parent.dk["всего"]) }
                    }
                }
                Pane {
                    visible: sec === "hw"; Layout.fillWidth: true
                    RowLayout { Layout.fillWidth: true; H2 { text: tx("Настройка оборудования"); Layout.fillWidth: true }
                        Btn { size: "sm"; kind: "primary"; text: tx("Сохранить")
                              onClicked: { const ok = B.saveConf({mem_cap_pct: hwCap.text, headroom_mib: hwHead.text, tier1_params: hwT1.text, bf16x3: hwBf.checked ? "1" : "0", prefetch: hwPf.checked ? "1" : "0"})
                                           toast(ok ? tx("Сохранено") : tx("Не удалось сохранить"), ok ? tx("Действует со следующего запуска движка.") : stv.confPath, ok ? "ok" : "bad"); stv.loadCfg() } } }
                    SRow { label: tx("Потолок памяти, %"); tip: tx("Доля памяти всей машины, выше которой движок не занимает ОЗУ и видеопамять. Допустимы значения от 50 до 99, по умолчанию 98. При нехватке ОЗУ обучение сохраняет снимок весов и останавливается."); Field { id: hwCap; Layout.preferredWidth: 200 } }
                    SRow { label: tx("Запас видеопамяти, МиБ"); tip: tx("Видеопамять, резервируемая под промежуточные данные шага.") + " " + tx("Пустое поле — значение из осмотра карты движком."); Field { id: hwHead; ph: tx("пусто — автоматически"); Layout.preferredWidth: 200 } }
                    SRow { label: tx("Бюджет среза строк"); tip: tx("Количество обучаемых чисел в видеопамяти: вес и два момента Adam, 12 байт на число.") + " " + tx("Пустое поле — значение из осмотра карты движком."); Field { id: hwT1; ph: tx("пусто — автоматически"); Layout.preferredWidth: 200 } }
                    SRow { label: "FP16×3 / BF16×3"; tip: tx("Сначала применяется FP16×3; если проба не пройдена, используется BF16×3 при успешной пробе, иначе FP32."); Sw { id: hwBf; checked: stv.hw["bf16x3"] === "1"; onToggled: (v) => checked = v } }
                    SRow { label: tx("Опережающая подкачка следующего слоя"); tip: tx("Следующий слой передаётся по шине, пока считается текущий."); Sw { id: hwPf; checked: (stv.hw["prefetch"] || "1") !== "0"; onToggled: (v) => checked = v } }
                }
                Pane {
                    visible: sec === "hw"; Layout.fillWidth: true
                    H2 { text: tx("Движок") }
                    SRow { label: tx("Состояние"); Pill { kind: hh && hh["движок"] ? "ok" : "warn"; text: hh && hh["движок"] ? tx("Собран") + (hh["гпу"] && hh["гпу"]["архитектура"] ? " · " + hh["гпу"]["архитектура"] : "") : hh && hh["движок_есть"] ? tx("Собран из прежней версии") : tx("Не собран") } }
                    SRow { label: tx("Сборка"); tip: tx("Занимает несколько минут.")
                        Btn { icon: "build"; text: tx("Пересобрать"); on: !!(hh && hh["компилятор"]) && !(hh && hh["настройка"]); onClicked: { B.setupRun("движок"); go("setup") } }
                        Small { text: hh && hh["компилятор"] ? hh["компилятор"] : tx("Компилятор отсутствует") } }
                }

                // вт5
                Pane {
                    visible: sec === "paths"; Layout.fillWidth: true
                    RowLayout { Layout.fillWidth: true; H2 { text: tx("Расположение файлов"); Layout.fillWidth: true }
                        Btn { size: "sm"; kind: "primary"; text: tx("Сохранить"); onClicked: { B.saveSettings({MODEL: pModel.text, WORK: pWork.text, HF_REPO: pRepo.text, HF_TOKEN: pTok.text.trim()}); toast(tx("Сохранено"), "", "ok"); B.refresh() } } }
                    SRow { wide: true; label: tx("Папка модели"); tip: tx("Расположение файла model.safetensors. Используется только для чтения.")
                        Field { id: pModel; Layout.fillWidth: true; mono: true }
                        Btn { size: "sm"; icon: "folder"; text: tx("Выбрать"); onClicked: { const p = B.pick("папка"); if (p) pModel.text = p } } }
                    SRow { wide: true; label: tx("Рабочая папка"); tip: tx("Журналы, токены, обученные веса.")
                        Field { id: pWork; Layout.fillWidth: true; mono: true }
                        Btn { size: "sm"; icon: "folder"; text: tx("Выбрать"); onClicked: { const p = B.pick("папка"); if (p) pWork.text = p } } }
                    SRow { wide: true; label: tx("Источник загрузки весов"); tip: "Hugging Face"; Field { id: pRepo; Layout.fillWidth: true; mono: true } }
                    SRow { wide: true; label: tx("Ключ Hugging Face"); tip: tx("Требуется для закрытых моделей (Llama, Gemma): сначала даётся согласие на странице модели, затем создаётся ключ с правом чтения (huggingface.co/settings/tokens). Ключ хранится в настройках программы и доступен только текущему пользователю.")
                        Field { id: pTok; Layout.fillWidth: true; mono: true; ph: "hf_…"; input.echoMode: input.activeFocus ? TextInput.Normal : TextInput.Password }
                        Btn { size: "sm"; icon: "open"; text: tx("Получить"); onClicked: Qt.openUrlExternally("https://huggingface.co/settings/tokens") } }
                }
                Pane {
                    visible: sec === "paths"; Layout.fillWidth: true
                    H2 { text: tx("Программа") }
                    Repeater {
                        model: stv.paths || []
                        delegate: SRow { required property var modelData; wide: true; label: modelData[0]
                            TextEdit { Layout.fillWidth: true; text: modelData[1]; readOnly: true; selectByMouse: true; wrapMode: TextEdit.WrapAnywhere; color: th.ink3; font.family: "monospace"; font.pixelSize: th.fs - 1.5; selectionColor: th.wash2 } }
                    }
                }

                // вт6
                Pane {
                    visible: sec === "keys"; Layout.fillWidth: true
                    H2 { text: tx("Клавиши") }
                    Repeater {
                        model: [["Ctrl+K", tx("Команды и поиск")], ["Ctrl+1 … Ctrl+6", tx("Разделы по порядку")], ["Ctrl+,", tx("Настройки")], ["Ctrl+Enter", tx("Начать обучение")],
                                ["Ctrl+Shift+L", tx("Сменить тему")], ["1 … 6", tx("Плитки графиков на «Ходе обучения»")], ["J", tx("Журнал запуска")], ["/", tx("Поиск по запускам")],
                                ["Esc", tx("Закрыть окно или диалог")], ["F11", tx("Во весь экран")], ["Ctrl+Q", tx("Завершить работу программы")]]
                        delegate: SRow {
                            required property var modelData
                            label: modelData[1]
                            Row {
                                spacing: 4
                                Repeater {
                                    model: modelData[0].split(/(\+| … )/)
                                    delegate: Item {
                                        required property string modelData
                                        width: modelData === "+" || modelData === " … " ? kp.implicitWidth : kk.width; height: 22
                                        Txt { id: kp; visible: modelData === "+" || modelData === " … "; anchors.centerIn: parent; text: modelData.trim(); color: th.ink3; font.pixelSize: th.fs - 2 }
                                        Rectangle { id: kk; visible: !kp.visible; width: Math.max(22, kt2.implicitWidth + 12); height: 22; radius: 5; color: th.fill2; border.width: 1; border.color: th.edge2
                                                    Txt { id: kt2; anchors.centerIn: parent; text: modelData; font.pixelSize: th.fs - 2; font.family: "monospace" } }
                                    }
                                }
                            }
                        }
                    }
                }

                // вт7
                Pane {
                    visible: sec === "about"; Layout.fillWidth: true
                    RowLayout {
                        spacing: 16
                        Logo { ink: th.ink; accent: th.accent; Layout.preferredWidth: 64; Layout.preferredHeight: 64 }
                        ColumnLayout { spacing: 2; Layout.fillWidth: true
                            Txt { text: "TierLM " + APPVER; font.pixelSize: th.fs + 9; font.weight: th.wth }
                            Txt { text: tx("Дообучение моделей, которые не помещаются в видеопамять"); Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone; Layout.topMargin: 4 }
                            Txt { text: tx("Веса хранятся в ОЗУ и подаются на видеокарту послойно с опережением, поэтому модель 12B в BF16 обучается на 8 ГБ. Изменяются собственные веса модели, а не только адаптер. Поддерживаются AMD и NVIDIA."); color: th.ink2; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone } }
                    }
                }
                Pane {
                    visible: sec === "about"; Layout.fillWidth: true
                    H2 { text: tx("Связь с автором") }
                    Repeater {
                        model: [["Telegram", "@So_bad_catty", "https://t.me/So_bad_catty"], [tx("ВКонтакте"), "Eberhard Gottman", ""], ["Discord", "so_bad_cat", ""]]
                        delegate: SRow {
                            required property var modelData
                            label: modelData[0]
                            Row {
                                spacing: 6
                                Btn { kind: "quiet"; size: "sm"; icon: "copy"; text: modelData[1]; tip: tx("Скопировать"); onClicked: { B.copy(modelData[1]); toast(tx("Скопировано"), modelData[1], "ok") } }
                                Btn { visible: modelData[2] !== ""; kind: "quiet"; size: "sm"; icon: "open"; tip: tx("Открыть в браузере"); onClicked: B.openUrl(modelData[2]) }
                            }
                        }
                    }
                    Small { text: tx("Шрифты") + ": Inter, Golos Text, Onest, Manrope, Jost, Geologica — SIL Open Font License 1.1"; Layout.topMargin: 6; Layout.fillWidth: true; wrapMode: Text.Wrap }
                }
            }
        }
    }
    function pickBg() { const p = B.pick("картинка"); if (!p) return; B.saveSettings({BG: p}); bgPath = p; apSet({bg: "image"}) }

    // вф1
    Item {
        id: winBox
        property bool open: false
        anchors.fill: parent; z: 600; visible: open
        onOpenChanged: if (!open) B.closeWin()
        Rectangle { anchors.fill: parent; color: th.scrim; MouseArea { anchors.fill: parent; onClicked: winBox.open = false } }
        Rectangle {
            id: wbox
            anchors.centerIn: parent
            width: Math.min(1180, parent.width - 60); height: parent.height - 60
            radius: th.r + 2; color: th.light ? "#fbfbfd" : "#232328"; border.width: 1; border.color: th.edge2
            scale: winBox.open ? 1 : .96; opacity: winBox.open ? 1 : 0
            Behavior on scale { enabled: th.motion; NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
            MouseArea { anchors.fill: parent }
            RowLayout {
                id: whd; x: 22; y: 16; width: parent.width - 44; spacing: 8
                H2 { text: wv ? wv.title || "" : "" }
                Icon { visible: !!(wv && wv.note); name: "info"; color: th.ink4; Layout.preferredWidth: 14; Layout.preferredHeight: 14; Tip { text: wv ? wv.note || "" : "" } }
                Small { text: wv ? wv.sub || "" : ""; Layout.fillWidth: true }
                Btn { visible: !!(wv && wv.kind === "log"); size: "sm"; kind: "quiet"; icon: "copy"; text: tx("Копировать вывод"); onClicked: { B.copy(wv.out || ""); toast(tx("Скопировано"), "", "ok") } }
                Btn { kind: "quiet"; icon: "x"; tip: tx("Закрыть") + " · Esc"; onClicked: winBox.open = false }
            }
            Scroll {
                anchors.fill: parent; anchors.topMargin: whd.y + whd.height + 12; anchors.leftMargin: 22; anchors.rightMargin: 10; anchors.bottomMargin: 8
                GridLayout {
                    visible: !!(wv && wv.kind === "chart"); Layout.fillWidth: true; columns: Math.max(2, Math.min(6, Math.floor(width / 170))); columnSpacing: 8; rowSpacing: 8
                    Repeater { model: ((wv && wv.stats ? wv.stats : []) || []).length; delegate: Tile { required property int index; readonly property var modelData: ((wv && wv.stats ? wv.stats : []) || [])[index] || []; Layout.fillWidth: true; label: modelData[0]; value: modelData[1]; note: modelData[2] } }
                }
                Repeater {
                    model: ((wv && wv.kind === "chart" ? wv.charts : []) || []).length
                    delegate: ColumnLayout {
                        readonly property var modelData: ((wv && wv.kind === "chart" ? wv.charts : []) || [])[index] || []
                        required property int index
                        Layout.fillWidth: true; spacing: 6; Layout.topMargin: 8
                        RowLayout { Txt { text: modelData.title; font.weight: th.wth }
                            Icon { visible: modelData.note !== ""; name: "info"; color: th.ink4; Layout.preferredWidth: 14; Layout.preferredHeight: 14; Tip { text: modelData.note } } }
                        Legend { visible: modelData.keys.length > 0; Layout.fillWidth: true; items: modelData.keys }
                        Plot { Layout.fillWidth: true; Layout.preferredHeight: index === 0 ? 300 : 200; key: "win:" + index }
                    }
                }
                Table { visible: !!(wv && wv.kind === "chart" && wv.rows && wv.rows.length); Layout.fillWidth: true; Layout.topMargin: 8; cols: wv && wv.cols ? wv.cols : []; rows: wv && wv.rows ? wv.rows : [] }
                // вф2
                Table {
                    visible: !!(wv && wv.kind === "log"); Layout.fillWidth: true; cols: [tx("Время"), tx("Событие"), tx("Что произошло")]; weights: [.16, .13, .71]; wrap: true
                    rows: wv && wv.events ? wv.events.map(e => [e.when, e.event, e.text]) : []
                }
                Txt { visible: !!(wv && wv.kind === "log"); text: tx("Вывод программы"); font.weight: th.wth; Layout.topMargin: 10 }
                Rectangle {
                    visible: !!(wv && wv.kind === "log"); Layout.fillWidth: true; radius: th.r2; color: th.field; border.width: 1; border.color: th.edge
                    implicitHeight: lout.implicitHeight + 20
                    TextEdit { id: lout; x: 10; y: 10; width: parent.width - 20; text: wv && wv.out ? wv.out : tx("Вывод отсутствует"); readOnly: true; selectByMouse: true; wrapMode: TextEdit.WrapAnywhere
                               color: th.ink2; font.family: "monospace"; font.pixelSize: th.fs - 1.5; selectionColor: th.wash2 }
                }
                // вф3
                Table { visible: !!(wv && wv.kind === "plan"); Layout.fillWidth: true; cols: [tx("Параметр"), tx("Значение"), tx("Источник")]; weights: [.24, .26, .5]; wrap: true; rows: wv && wv.rows && wv.kind === "plan" ? wv.rows : [] }
                Small { visible: !!(wv && wv.kind === "plan" && wv.note); text: wv ? wv.note || "" : "" }
                Txt { visible: !!(wv && wv.kind === "plan" && wv.env); text: tx("Переменные окружения движка"); font.weight: th.wth; Layout.topMargin: 10 }
                Rectangle {
                    visible: !!(wv && wv.kind === "plan" && wv.env); Layout.fillWidth: true; radius: th.r2; color: th.field; border.width: 1; border.color: th.edge
                    implicitHeight: envt.implicitHeight + 20
                    TextEdit { id: envt; x: 10; y: 10; width: parent.width - 20; text: wv && wv.env ? wv.env : ""; readOnly: true; selectByMouse: true; wrapMode: TextEdit.WrapAnywhere
                               color: th.ink2; font.family: "monospace"; font.pixelSize: th.fs - 1.5; selectionColor: th.wash2 }
                }
            }
        }
    }

    component Table: ColumnLayout {
        id: tbl
        property var cols: []
        property var rows: []
        property var weights: []
        property bool wrap: false
        spacing: 0
        function w(i) { const n = cols.length; if (weights.length === n) return weights[i]; return 1 / n }
        Item {
            Layout.fillWidth: true; implicitHeight: 30
            Row { anchors.fill: parent; Repeater { model: tbl.cols; delegate: Small { required property string modelData; required property int index; width: tbl.width * tbl.w(index); height: 30; text: modelData; rightPadding: 10 } } }
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep }
        }
        Repeater {
            model: (tbl.rows || []).length
            delegate: Item {
                id: tr
                required property int index
                readonly property var cells: (tbl.rows || [])[index] || []
                Layout.fillWidth: true; implicitHeight: Math.max(Math.round(30 * th.d), trow.implicitHeight + 12)
                Row { id: trow; y: 6; width: parent.width
                    Repeater { model: tbl.cols.length; delegate: Txt { required property int index; width: tbl.width * tbl.w(index); rightPadding: 10; text: tr.cells[index] === undefined ? "" : String(tr.cells[index]); font.pixelSize: th.fs - .5
                               wrapMode: tbl.wrap ? Text.Wrap : Text.NoWrap; elide: tbl.wrap ? Text.ElideNone : Text.ElideRight; color: index === 0 ? th.ink2 : th.ink } } }
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: Math.max(1, th.hair); color: th.sep; opacity: .7 }
            }
        }
    }

    // вц1
    function ask(o, cb) { dlg.show(o, cb) }
    Item {
        id: dlg
        anchors.fill: parent; z: 800; visible: false
        property var o: ({})
        property var cb: null
        // вч1
        readonly property bool stale: visible && !!o.liveRun && !(tv && tv.live && tv.name === o.liveRun)
        onStaleChanged: if (stale) Qt.callLater(() => { if (dlg.stale) dlg.fin(false) })
        function show(o, cb) { dlg.o = o; dlg.cb = cb; dIn.text = o.input && o.input.value ? o.input.value : ""; dArea.text = o.input && o.input.value ? o.input.value : ""; visible = true
                               Qt.callLater(() => { if (o.input && o.input.area) dArea.edit.forceActiveFocus(); else if (o.input) dIn.input.forceActiveFocus(); else dOk.forceActiveFocus() }) }
        function fin(r) { visible = false; const c = cb; cb = null; if (c) c(r) }
        Rectangle { anchors.fill: parent; color: th.scrim; MouseArea { anchors.fill: parent; onClicked: dlg.fin(false) } }
        Rectangle {
            anchors.centerIn: parent; width: Math.min(460, parent.width - 40); implicitHeight: dcol.implicitHeight + 40; height: implicitHeight
            radius: th.r + 2; color: th.light ? "#fbfbfd" : "#26262b"; border.width: 1; border.color: th.edge2
            MouseArea { anchors.fill: parent }
            ColumnLayout {
                id: dcol; x: 22; y: 20; width: parent.width - 44; spacing: 12
                Txt { text: dlg.o.title || ""; font.pixelSize: th.fs + 2; font.weight: th.wth; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone }
                Txt { visible: !!dlg.o.text; text: dlg.o.text || ""; color: th.ink2; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone }
                Field { id: dIn; visible: !!(dlg.o.input && !dlg.o.input.area); Layout.fillWidth: true; ph: dlg.o.input ? dlg.o.input.ph || "" : ""; onAccepted: dlg.fin(text) }
                Area { id: dArea; visible: !!(dlg.o.input && dlg.o.input.area); Layout.fillWidth: true; minH: 150; mono: true; ph: dlg.o.input ? dlg.o.input.ph || "" : "" }
                RowLayout {
                    Layout.fillWidth: true; Layout.topMargin: 4; spacing: 8
                    Item { Layout.fillWidth: true }
                    Btn { kind: "quiet"; text: dlg.o.cancel || tx("Отмена"); onClicked: dlg.fin(false) }
                    Btn { id: dOk; kind: dlg.o.danger ? "solidDanger" : "primary"; text: dlg.o.ok || tx("Готово")
                          onClicked: dlg.fin(dlg.o.input ? (dlg.o.input.area ? dArea.text : dIn.text) : true) }
                }
            }
        }
    }

    // вш1
    Item {
        id: menu
        anchors.fill: parent; z: 850; visible: false
        property Item owner: null
        property var opts: []
        property var cur
        property var cb: null
        function openFor(it, opts, cur, cb) {
            owner = it; menu.opts = opts; menu.cur = cur; menu.cb = cb
            const p = it.mapToItem(menu, 0, 0)
            const h = Math.min(320, opts.length * 32 + 8)
            mbox.width = Math.max(it.width, 180)
            mbox.x = Math.min(p.x, menu.width - mbox.width - 8)
            mbox.y = p.y + it.height + 4 + h > menu.height - 8 ? p.y - h - 4 : p.y + it.height + 4
            mbox.height = h
            visible = true
        }
        function close() { visible = false; owner = null }
        MouseArea { anchors.fill: parent; onClicked: menu.close(); onWheel: (w) => menu.close() }
        Rectangle {
            id: mbox; radius: th.rc + 2; color: th.light ? "#ffffff" : "#2c2c32"; border.width: 1; border.color: th.edge2
            ListView {
                id: mlist; anchors.fill: parent; anchors.margins: 4; model: menu.opts; clip: true; boundsBehavior: Flickable.StopAtBounds
                delegate: Rectangle {
                    required property var modelData
                    width: mlist.width; height: 32; radius: th.rc
                    color: mim.containsMouse ? th.fill2 : String(modelData[0]) === String(menu.cur) ? th.wash : "transparent"
                    Txt { x: 10; width: parent.width - 34; anchors.verticalCenter: parent.verticalCenter; text: modelData[1]; color: String(modelData[0]) === String(menu.cur) ? th.accentInk : th.ink }
                    Icon { visible: String(modelData[0]) === String(menu.cur); name: "check"; color: th.accent; width: 13; height: 13; anchors.right: parent.right; anchors.rightMargin: 10; anchors.verticalCenter: parent.verticalCenter }
                    MouseArea { id: mim; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { const c = menu.cb, v = modelData[0]; menu.close(); if (c) c(v) } }
                }
            }
        }
    }

    // вэ1
    Item {
        id: cmdp
        anchors.fill: parent; z: 820; visible: false
        property var all: []
        property var shown: []
        property int sel: 0
        function open() { all = commands(); pin.text = ""; filter(); visible = true; Qt.callLater(() => pin.input.forceActiveFocus()) }
        function close() { visible = false }
        function score(t, q) { t = t.toLowerCase(); if (!q) return 1; if (t.indexOf(q) >= 0) return 3 - (t.indexOf(q) > 0 ? 1 : 0); let i = 0; for (const c of t) { if (c === q[i]) i++; if (i === q.length) return 1 } return 0 }
        function filter() { const q = pin.text.trim().toLowerCase(); shown = q ? all.map(c => ({c: c, s: score(c.t, q)})).filter(x => x.s > 0).sort((a, b) => b.s - a.s).map(x => x.c) : all; sel = 0 }
        function run(i) { const c = shown[i]; if (!c) return; close(); Qt.callLater(c.f) }
        Rectangle { anchors.fill: parent; color: th.scrim; MouseArea { anchors.fill: parent; onClicked: cmdp.close() } }
        Rectangle {
            x: (parent.width - width) / 2; y: Math.max(40, parent.height * .12); width: Math.min(620, parent.width - 40); height: Math.min(460, 62 + plist.contentHeight)
            radius: th.r + 2; color: th.light ? "#fbfbfd" : "#26262b"; border.width: 1; border.color: th.edge2
            MouseArea { anchors.fill: parent }
            Field {
                id: pin; x: 10; y: 10; width: parent.width - 20; icon: "search"; ph: tx("Команда, раздел или запуск")
                onEdited: cmdp.filter()
                input.Keys.onPressed: (e) => {
                    if (e.key === Qt.Key_Down) { cmdp.sel = Math.min(cmdp.shown.length - 1, cmdp.sel + 1); plist.positionViewAtIndex(cmdp.sel, ListView.Contain); e.accepted = true }
                    else if (e.key === Qt.Key_Up) { cmdp.sel = Math.max(0, cmdp.sel - 1); plist.positionViewAtIndex(cmdp.sel, ListView.Contain); e.accepted = true }
                    else if (e.key === Qt.Key_Return || e.key === Qt.Key_Enter) { cmdp.run(cmdp.sel); e.accepted = true }
                    else if (e.key === Qt.Key_Escape) { cmdp.close(); e.accepted = true }
                }
            }
            ListView {
                id: plist; x: 8; y: 52; width: parent.width - 16; height: parent.height - 60; clip: true; model: cmdp.shown; boundsBehavior: Flickable.StopAtBounds
                section.property: "g"; section.criteria: ViewSection.FullString
                section.delegate: Small { required property string section; text: section; height: 26; leftPadding: 10; font.weight: th.wth; font.pixelSize: th.fs - 2.5 }
                delegate: Rectangle {
                    required property var modelData
                    required property int index
                    width: plist.width; height: 34; radius: th.rc
                    color: index === cmdp.sel ? th.accent : pim.containsMouse ? th.fill2 : "transparent"
                    RowLayout { anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10; spacing: 10
                        Icon { name: modelData.i || "arrow"; color: index === cmdp.sel ? th.onAccent : th.ink3; Layout.preferredWidth: 15; Layout.preferredHeight: 15 }
                        Txt { text: modelData.t; Layout.fillWidth: true; color: index === cmdp.sel ? th.onAccent : th.ink }
                        Small { visible: !!modelData.s; text: modelData.s || ""; color: index === cmdp.sel ? th.onAccent : th.ink3 }
                        Small { visible: !!modelData.k; text: modelData.k || ""; color: index === cmdp.sel ? th.onAccent : th.ink3; font.family: "monospace" } }
                    MouseArea { id: pim; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: cmdp.run(index) }
                }
                WheelHandler { onWheel: (ev) => { plist.contentY = Math.max(plist.originY, Math.min(plist.originY + plist.contentHeight - plist.height, plist.contentY - ev.angleDelta.y / 2)) } }
            }
            Small { visible: cmdp.shown.length === 0; x: 20; y: 60; text: tx("Ничего не найдено") }
        }
    }
    function commands() {
        const C = []
        const V = [["setup", tx("Новое обучение"), "setup"], ["train", tx("Ход обучения"), "train"], ["runs", tx("Запуски"), "runs"], ["chat", tx("Разговор"), "chat"], ["bench", tx("Проверка"), "bench"], ["models", tx("Модели"), "chip"], ["settings", tx("Настройки"), "settings"]]
        V.forEach((v, i) => C.push({g: tx("Разделы"), t: v[1], i: v[2], k: "Ctrl+" + (i + 1), f: () => go(v[0])}))
        secs.forEach(s => C.push({g: tx("Настройки"), t: tx("Настройки") + ": " + s[1], i: s[2], f: () => { sec = s[0]; go("settings") }}))
        if (sv && sv.launch && sv.launch.can) C.push({g: tx("Действия"), t: tx("Начать обучение"), i: "play", k: "Ctrl+Enter", f: () => B.launch()})
        if (liveName) C.push({g: tx("Действия"), t: tx("Остановить") + " «" + liveName + "»", i: "stop", f: () => { B.openRun(liveName); go("train"); askStop() }})
        C.push({g: tx("Действия"), t: th.light ? tx("Тёмная тема") : tx("Светлая тема"), i: "theme", k: "Ctrl+Shift+L", f: () => apSet({theme: th.light ? "dark" : "light"})})
        C.push({g: tx("Действия"), t: en ? "Русский язык" : "English", i: "globe", f: () => B.setLang(en ? "ru" : "en")})
        if (hh && hh["компилятор"]) C.push({g: tx("Действия"), t: tx("Собрать движок"), i: "build", f: () => { B.setupRun("движок"); go("setup") }})
        C.push({g: tx("Действия"), t: tx("Завершить работу программы"), i: "power", k: "Ctrl+Q", f: killAll})
        presets.forEach(p => C.push({g: tx("Оформление"), t: tx("Набор") + " «" + p[0] + "»", i: "palette", f: () => applyPreset(p[1])}))
        ;(ui.presets || []).forEach(p => C.push({g: tx("Оформление"), t: tx("Собственный набор") + " «" + p.n + "»", i: "palette", f: () => applyPreset(p.a)}))
        ;(hh && hh["запуски"] ? hh["запуски"] : []).forEach(r => {
            C.push({g: tx("Запуски"), t: tx("Открыть") + " «" + r["имя"] + "»", i: "open", s: r["состояние"] === "идёт" ? tx("Выполняется") : "", f: () => { B.openRun(r["имя"]); go("train") }})
            if (r["веса"]) C.push({g: tx("Запуски"), t: tx("Разговор с") + " «" + r["имя"] + "»", i: "chat", f: () => { chatView.pick(r["имя"]); go("chat") }})
        })
        return C
    }
    function killAll() {
        ask({title: tx("Завершить работу программы?"), text: liveName ? tx("Обучение") + " «" + liveName + "» " + tx("будет остановлено, программа будет закрыта полностью.") : tx("Программа будет закрыта полностью."), ok: tx("Завершить"), danger: true},
            (r) => { if (r) B.kill() })
    }

    // га1
    function toast(t, x, k) { toasts.add(t, x, k) }
    Connections { target: B; function onToast(t, x, k) { toasts.add(t, x, k) } function onGo_train() { go("train") } }
    Column {
        id: toasts
        anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 22; spacing: 8; z: 900
        property int seq: 0
        function add(t, x, k) {
            for (let i = 0; i < tmodel.count; i++) if (tmodel.get(i).head === t && tmodel.get(i).body === (x || "")) return     // гв1
            if (tmodel.count >= 4) tmodel.remove(0)
            tmodel.append({head: t, body: x || "", kind: k || "info", uid: ++seq})
        }
        function drop(uid) { for (let i = 0; i < tmodel.count; i++) if (tmodel.get(i).uid === uid) { tmodel.remove(i); return } }
        ListModel { id: tmodel }
        Repeater {
            model: tmodel
            delegate: Rectangle {
                id: tst
                required property string head
                required property string body
                required property string kind
                required property int uid
                width: Math.min(400, Math.max(250, tcol2.implicitWidth + 50)); height: tcol2.implicitHeight + 22
                radius: th.r2 + 2; color: th.light ? "#ffffff" : "#2e2e34"; border.width: 1; border.color: th.edge2
                opacity: 0
                Component.onCompleted: opacity = 1
                Behavior on opacity { NumberAnimation { duration: 180 } }
                Timer { interval: tst.kind === "bad" ? 9000 : 4400; running: true; onTriggered: tst.gone() }
                function gone() { opacity = 0; rm.start() }
                Timer { id: rm; interval: 200; onTriggered: toasts.drop(tst.uid) }
                RowLayout {
                    x: 14; y: 11; width: parent.width - 28; spacing: 10
                    Icon { name: tst.kind === "ok" ? "ok" : tst.kind === "bad" ? "bad" : tst.kind === "warn" ? "warn" : "info"; color: tst.kind === "ok" ? th.ok : tst.kind === "bad" ? th.bad : tst.kind === "warn" ? th.warn : th.accent
                           Layout.preferredWidth: 16; Layout.preferredHeight: 16; Layout.alignment: Qt.AlignTop }
                    ColumnLayout { id: tcol2; spacing: 2; Layout.fillWidth: true
                        Txt { text: tst.head; font.weight: th.wth; Layout.fillWidth: true; wrapMode: Text.Wrap; elide: Text.ElideNone }
                        Txt { visible: tst.body !== ""; text: tst.body; color: th.ink2; font.pixelSize: th.fs - 1; Layout.fillWidth: true; Layout.maximumWidth: 330; wrapMode: Text.WrapAnywhere; elide: Text.ElideNone } }
                }
                MouseArea { anchors.fill: parent; onClicked: tst.gone() }
            }
        }
    }

    // гг1
    Item {
        id: tipLayer
        anchors.fill: parent; z: 1000
        property Item tgt: null
        property string txt: ""
        property bool shown: false
        property bool low: false   // гг2
        function arm(it, t) { low = false; tgt = it; txt = t; shown = false; tipT.restart() }
        function armLow(it, t) { if (tgt && !low) return; low = true; tgt = it; txt = t; shown = false; tipT.restart() }
        function disarm(it) { if (tgt === it) { tipT.stop(); shown = false; tgt = null; low = false } }
        function disarmLow(it) { if (tgt === it && low) disarm(it) }
        function hideNow() { tipT.stop(); shown = false; tgt = null }
        Timer { id: tipT; interval: 450; onTriggered: { if (tipLayer.tgt) { tipLayer.place(); tipLayer.shown = true } } }
        function place() {
            const t = tgt; if (!t) return
            const p = t.mapToItem(tipLayer, 0, 0)
            const w = tipBox.width, h = tipBox.height
            let x = p.x + t.width / 2 - w / 2, y = p.y + t.height + 8
            if (y + h > height - 8) y = p.y - h - 8
            tipBox.x = Math.max(8, Math.min(width - w - 8, x)); tipBox.y = Math.max(8, y)
        }
        Rectangle {
            id: tipBox
            visible: tipLayer.shown && tipLayer.txt !== ""
            width: Math.min(360, tipTxt.implicitWidth + 22); height: tipTxt.height + 14
            radius: 8; color: th.pop; border.width: 1; border.color: th.edge2
            Txt { id: tipTxt; gloss: false; x: 11; y: 7; width: Math.min(338, implicitWidth); text: tipLayer.txt; color: th.ink2; font.pixelSize: th.fs - 1.5; wrapMode: Text.Wrap; elide: Text.ElideNone; lineHeight: 1.1 }
        }
    }

    // гд1
    Shortcut { sequences: ["Ctrl+K", "Ctrl+P"]; onActivated: cmdp.visible ? cmdp.close() : cmdp.open() }
    Shortcut { sequence: "Ctrl+Q"; onActivated: killAll() }
    Shortcut { sequence: "Ctrl+Shift+L"; onActivated: apSet({theme: th.light ? "dark" : "light"}) }
    Shortcut { sequence: "Ctrl+,"; onActivated: go("settings") }
    Shortcut { sequences: ["Ctrl+Return", "Ctrl+Enter"]; enabled: view === "setup"; onActivated: B.launch() }
    Shortcut { sequence: "Ctrl+1"; onActivated: go("setup") }
    Shortcut { sequence: "Ctrl+2"; onActivated: go("train") }
    Shortcut { sequence: "Ctrl+3"; onActivated: go("runs") }
    Shortcut { sequence: "Ctrl+4"; onActivated: go("chat") }
    Shortcut { sequence: "Ctrl+5"; onActivated: go("bench") }
    Shortcut { sequence: "Ctrl+6"; onActivated: go("settings") }
    Shortcut { sequence: "F11"; onActivated: win.visibility === Window.FullScreen ? win.showNormal() : win.showFullScreen() }
    Shortcut {
        sequence: "Esc"
        onActivated: { if (menu.visible) menu.close(); else if (cmdp.visible) cmdp.close(); else if (dlg.visible) dlg.fin(false); else if (winBox.open) winBox.open = false }
    }
    Shortcut { sequence: "J"; enabled: !typing && view === "train" && !winBox.open; onActivated: openWin("log") }
    Shortcut { sequence: "/"; enabled: !typing && view === "runs"; onActivated: runsView.focusSearch() }
    Repeater {
        model: 6
        delegate: Item {
            required property int index
            Shortcut { sequence: String(index + 1); enabled: !typing && view === "train" && !winBox.open && index < chartOrder.length; onActivated: openWin(chartOrder[index]) }
        }
    }

    Component.onCompleted: {
        let v = ui.start || "auto"
        if (v === "auto") v = hh && hh["идёт"] ? "train" : "setup"
        go(v)
    }
}
