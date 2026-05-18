/**
 * inspect_benchmark.groovy — benchmark result + gold standard overlay
 *
 * Usage:
 *   Fiji → Plugins → Scripting → Script Editor → Language: Groovy
 *   Edit CONFIG → Run
 *
 * 양쪽 SWC가 같은 µm 좌표계를 쓰려면:
 *   - ours SWC    : step3 출력, 이미 µm (resampled stack 기준)
 *   - gold SWC    : 원본 픽셀 단위 → VOXEL_XY / VOXEL_Z 곱해서 µm 변환
 */

import ij.*
import ij.gui.*
import ij.process.*
import java.awt.Color
import java.io.File

// ── CONFIG ──────────────────────────────────────────────────────
def SAMPLE       = "neuron2"
def ROOT         = "/Users/lee/Tracer/benchmark"

def STACK_PATH   = "${ROOT}/methods/ours/output/${SAMPLE}/stack_preprocessed.tif"
def OUR_SWC      = "${ROOT}/results/ours/${SAMPLE}.swc"
def GOLD_SWC     = "${ROOT}/data/gold_standard/${SAMPLE}.swc"  // "" 로 비활성화

// preprocessed stack 의 voxel 크기 (µm)
def VOXEL_ISO    = 0.5377f

// gold standard 원본 voxel 크기 (pixelsize.txt 참고)
def GOLD_VOXEL_XY = 0.2688637f
def GOLD_VOXEL_Z  = 1.0f

def SHOW_GOLD    = true   // gold standard 오버레이 표시 여부

// soma (soma.json centroid_vox * voxel_iso)
def SOMA_X_UM    = 523.02f * VOXEL_ISO
def SOMA_Y_UM    = 522.06f * VOXEL_ISO
def SOMA_Z_UM    = 128.57f * VOXEL_ISO
def SOMA_R_UM    = 6.23f

def LINE_WIDTH   = 2.0f
def SLICE_THICK  = 3

// ── COLORS ──────────────────────────────────────────────────────
def PALETTE_OURS = [
    new Color(31,119,180),  new Color(255,127,14),
    new Color(44,160,44),   new Color(214,39,40),
    new Color(148,103,189), new Color(140,86,75),
    new Color(227,119,194), new Color(127,127,127),
]
def COLOR_GOLD = new Color(255, 220, 50)   // gold = 노란색

// ── SWC LOADER ──────────────────────────────────────────────────
def loadSwc = { String path, float sx, float sy, float sz ->
    // sx/sy/sz: scale factors to convert file coords → µm
    def nodes = [:]
    new File(path).eachLine { line ->
        if (line.startsWith('#') || line.trim().isEmpty()) return
        def p = line.trim().split("\\s+")
        if (p.size() < 7) return
        nodes[p[0].toInteger()] = [
            x_um: p[2].toFloat() * sx,
            y_um: p[3].toFloat() * sy,
            z_um: p[4].toFloat() * sz,
            parent: p[6].toInteger()
        ]
    }
    nodes
}

// ── BUILD SEGMENTS ───────────────────────────────────────────────
def buildSegs = { nodes, palette, defaultColor ->
    def children = [:].withDefault { [] }
    nodes.each { nid, n -> if (n.parent != -1) children[n.parent] << nid }
    def primaryIds = (children[1] ?: []).sort()
    def primaryColor = [:]
    primaryIds.eachWithIndex { pid, i -> primaryColor[pid] = palette ? palette[i % palette.size()] : defaultColor }
    def primaryAncestor = { nid ->
        def cur = nid
        while (nodes.containsKey(cur) && nodes[cur].parent > 1) cur = nodes[cur].parent
        cur
    }
    def segs = []
    nodes.each { nid, n ->
        def par = n.parent
        if (par == -1 || !nodes.containsKey(par)) return
        def p = nodes[par]
        def pid = primaryAncestor(nid)
        def col = defaultColor ?: primaryColor.getOrDefault(pid, Color.WHITE)
        segs << [z1:(n.z_um/VOXEL_ISO).round() as int, z2:(p.z_um/VOXEL_ISO).round() as int,
                 x1f:n.x_um/VOXEL_ISO, y1f:n.y_um/VOXEL_ISO,
                 x2f:p.x_um/VOXEL_ISO, y2f:p.y_um/VOXEL_ISO, col:col]
    }
    segs
}

// ── SLICE MAP ────────────────────────────────────────────────────
def buildSliceMap = { segs, int nSlices ->
    def map = new HashMap<Integer, List>()
    segs.each { seg ->
        def zMin = Math.min(seg.z1, seg.z2)
        def zMax = Math.max(seg.z1, seg.z2)
        for (int z = Math.max(0, zMin - SLICE_THICK); z <= Math.min(nSlices-1, zMax + SLICE_THICK); z++) {
            def dz = (float)(seg.z2 - seg.z1)
            def x1, y1, x2, y2
            if (Math.abs(dz) > 0.5f) {
                def tLo = Math.max(0f, Math.min(1f, (z - SLICE_THICK - seg.z1) / dz))
                def tHi = Math.max(0f, Math.min(1f, (z + SLICE_THICK - seg.z1) / dz))
                x1 = seg.x1f + tLo*(seg.x2f - seg.x1f); y1 = seg.y1f + tLo*(seg.y2f - seg.y1f)
                x2 = seg.x1f + tHi*(seg.x2f - seg.x1f); y2 = seg.y1f + tHi*(seg.y2f - seg.y1f)
            } else { x1=seg.x1f; y1=seg.y1f; x2=seg.x2f; y2=seg.y2f }
            if (!map.containsKey(z)) map[z] = []
            map[z] << [x1:x1, y1:y1, x2:x2, y2:y2, col:seg.col]
        }
    }
    map
}

// ── MAIN ────────────────────────────────────────────────────────
println "Loading SWCs..."
def nodesOurs = loadSwc(OUR_SWC, 1.0f, 1.0f, 1.0f)           // 이미 µm
println "  ours: ${nodesOurs.size()} nodes"

def segsAll = buildSegs(nodesOurs, PALETTE_OURS, null)

if (SHOW_GOLD && new File(GOLD_SWC).exists()) {
    // gold standard: 픽셀 → µm 변환
    def nodesGold = loadSwc(GOLD_SWC, GOLD_VOXEL_XY, GOLD_VOXEL_XY, GOLD_VOXEL_Z)
    println "  gold: ${nodesGold.size()} nodes"
    segsAll += buildSegs(nodesGold, null, COLOR_GOLD)
}

println "Opening stack: ${STACK_PATH}"
def imp = IJ.openImage(STACK_PATH)
if (!imp) { IJ.error("Cannot open: ${STACK_PATH}"); return }
imp.show()
IJ.run(imp, "Enhance Contrast", "saturated=0.35")

def nSlices = imp.getNSlices()
println "  ${imp.getWidth()} x ${imp.getHeight()} x ${nSlices}  voxel=${VOXEL_ISO} µm"

def sliceMap = buildSliceMap(segsAll, nSlices)

def somaZpx = Math.round(SOMA_Z_UM / VOXEL_ISO) as int
def somaRpx = SOMA_R_UM / VOXEL_ISO

def drawOverlay = { int sliceIdx ->
    def z0 = sliceIdx - 1
    def ov = new Overlay()
    // soma circle
    def dz = Math.abs(z0 - somaZpx)
    if (dz <= somaRpx) {
        def r2 = Math.sqrt(Math.max(0.0, somaRpx*somaRpx - dz*dz))
        def ell = new OvalRoi((float)(SOMA_X_UM/VOXEL_ISO - r2), (float)(SOMA_Y_UM/VOXEL_ISO - r2),
                              (float)(r2*2), (float)(r2*2))
        ell.setStrokeColor(new Color(255, 80, 80)); ell.setStrokeWidth(2.0f)
        ov.add(ell)
    }
    // segments
    sliceMap.get(z0, []).each { s ->
        def roi = new Line((float)s.x1, (float)s.y1, (float)s.x2, (float)s.y2)
        roi.setStrokeWidth(LINE_WIDTH); roi.setStrokeColor(s.col); ov.add(roi)
    }
    imp.setOverlay(ov); imp.updateAndDraw()
}

drawOverlay(imp.getCurrentSlice())

def listener = [
    imageUpdated: { ImagePlus u -> if (u.is(imp)) drawOverlay(u.getCurrentSlice()) },
    imageOpened: { ImagePlus p -> }, imageClosed: { ImagePlus p -> }
] as ImageListener
ImagePlus.addImageListener(listener)

println ""
println "우리 결과 (색상) + gold standard (노란색) 오버레이"
println "Z 슬라이더로 이동  |  Ctrl+Shift+O: 오버레이 토글"
