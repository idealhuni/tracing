/**
 * inspect_fiji.groovy — FIJI interactive SWC overlay
 *
 * Usage:
 *   1. FIJI → Plugins → Scripting → Script Editor
 *   2. Language → Groovy
 *   3. File → Open → inspect_fiji.groovy
 *   4. Edit CONFIG below
 *   5. Run
 *
 * Controls:
 *   Z slider         : scroll slices
 *   Ctrl+Shift+O     : toggle overlay
 *   Image → B&C      : brightness/contrast
 */

import ij.*
import ij.gui.*
import ij.process.*
import java.awt.Color
import java.io.File

// ── CONFIG ──────────────────────────────────────────────────
def STACK_PATH   = "/Users/lee/Tracer/tracer_aniso/output/stack_preprocessed.tif"
def SWC_PATH     = "/Users/lee/Tracer/tracer_aniso/output/neurons_riem.swc"
def VOXEL_ISO    = 0.342f   // µm/voxel

def LINE_WIDTH   = 3.0f     // line thickness (pixels)
def SLICE_THICK  = 3        // ±N slices: show segment if it spans this range

// Soma: drawn as filled circle on each relevant slice
def SOMA_X_UM    = 150.0f   // soma centroid µm (update from soma.json)
def SOMA_Y_UM    = 160.0f
def SOMA_Z_UM    = 91.0f
def SOMA_R_UM    = 10.3f

// ── PALETTE ─────────────────────────────────────────────────
def PALETTE = [
    new Color(31,119,180),  new Color(255,127,14),
    new Color(44,160,44),   new Color(214,39,40),
    new Color(148,103,189), new Color(140,86,75),
    new Color(227,119,194), new Color(127,127,127),
    new Color(188,189,34),  new Color(23,190,207),
    new Color(174,199,232), new Color(255,187,120),
    new Color(152,223,138), new Color(255,152,150),
    new Color(197,176,213), new Color(196,156,148),
]

// ── LOAD SWC ────────────────────────────────────────────────
println "Loading SWC..."
def nodes = [:]
new File(SWC_PATH).eachLine { line ->
    if (line.startsWith('#') || line.trim().isEmpty()) return
    def p = line.trim().split("\\s+")
    nodes[p[0].toInteger()] = [
        x_um: p[2].toFloat(), y_um: p[3].toFloat(), z_um: p[4].toFloat(),
        parent: p[6].toInteger()
    ]
}
println "  ${nodes.size()} nodes"

// Children map + primary ancestors
def children = [:].withDefault { [] }
nodes.each { nid, n -> if (n.parent != -1) children[n.parent] << nid }

def primaryAncestor = { nid ->
    def cur = nid
    while (nodes.containsKey(cur) && nodes[cur].parent != -1 && nodes[cur].parent != 1)
        cur = nodes[cur].parent
    cur
}
def primaryIds = (children[1] ?: []).sort()
def primaryColor = [:]
primaryIds.eachWithIndex { pid, i -> primaryColor[pid] = PALETTE[i % PALETTE.size()] }

// ── PRECOMPUTE SEGMENTS (voxel coords) ──────────────────────
def segs = []
nodes.each { nid, n ->
    def par = n.parent
    if (par == -1 || !nodes.containsKey(par)) return
    def p = nodes[par]
    def pid = primaryAncestor(nid)
    def col = primaryColor.getOrDefault(pid, Color.WHITE)

    segs << [
        z1 : Math.round(n.z_um / VOXEL_ISO) as int,
        z2 : Math.round(p.z_um / VOXEL_ISO) as int,
        x1f: n.x_um / VOXEL_ISO,
        y1f: n.y_um / VOXEL_ISO,
        x2f: p.x_um / VOXEL_ISO,
        y2f: p.y_um / VOXEL_ISO,
        col: col
    ]
}
println "  ${segs.size()} segments"

// ── GROUP SEGMENTS BY SLICE ──────────────────────────────────
// segsForSlice[z] = list of [x1,y1,x2,y2,color] to draw at that slice
// For each segment spanning z1..z2, add to slices z1-THICK .. z2+THICK
// XY coords: interpolate at each slice z

def buildSliceMap = { int nSlices ->
    def map = new HashMap<Integer, List>()
    segs.each { seg ->
        def zMin = Math.min(seg.z1, seg.z2)
        def zMax = Math.max(seg.z1, seg.z2)
        def from = Math.max(0, zMin - SLICE_THICK)
        def to   = Math.min(nSlices-1, zMax + SLICE_THICK)

        for (int z = from; z <= to; z++) {
            // Interpolate XY at this Z
            def x1, y1, x2, y2
            def dz = (float)(seg.z2 - seg.z1)
            if (Math.abs(dz) > 0.5f) {
                // Find where segment crosses z and z±1 (draw as short line)
                def tMid = Math.max(0f, Math.min(1f, (z - seg.z1) / dz))
                def tLo  = Math.max(0f, Math.min(1f, (z - SLICE_THICK - seg.z1) / dz))
                def tHi  = Math.max(0f, Math.min(1f, (z + SLICE_THICK - seg.z1) / dz))
                x1 = seg.x1f + tLo * (seg.x2f - seg.x1f)
                y1 = seg.y1f + tLo * (seg.y2f - seg.y1f)
                x2 = seg.x1f + tHi * (seg.x2f - seg.x1f)
                y2 = seg.y1f + tHi * (seg.y2f - seg.y1f)
            } else {
                x1 = seg.x1f; y1 = seg.y1f
                x2 = seg.x2f; y2 = seg.y2f
            }
            if (!map.containsKey(z)) map[z] = []
            map[z] << [x1:x1, y1:y1, x2:x2, y2:y2, col:seg.col]
        }
    }
    map
}

// ── OPEN STACK ──────────────────────────────────────────────
println "Opening stack..."
def imp = IJ.openImage(STACK_PATH)
if (!imp) { IJ.error("Cannot open: ${STACK_PATH}"); return }
imp.show()
IJ.run(imp, "Enhance Contrast", "saturated=0.35")

def nSlices = imp.getNSlices()
println "  ${imp.getWidth()} x ${imp.getHeight()} x ${nSlices} slices"

// Build slice→segments map
println "Building slice map..."
def sliceMap = buildSliceMap(nSlices)
println "  Slices with segments: ${sliceMap.size()}"

// ── DRAW OVERLAY FOR CURRENT SLICE ──────────────────────────
def somaZpx = Math.round(SOMA_Z_UM / VOXEL_ISO) as int
def somaRpx = SOMA_R_UM / VOXEL_ISO

def drawOverlay = { int sliceIdx ->   // sliceIdx: 1-based
    def z0 = sliceIdx - 1             // 0-based
    def ov = new Overlay()

    // Soma circle
    def dz = Math.abs(z0 - somaZpx)
    if (dz <= somaRpx) {
        def r_slice = Math.sqrt(Math.max(0.0, somaRpx*somaRpx - dz*dz))
        def sx = SOMA_X_UM / VOXEL_ISO - r_slice
        def sy = SOMA_Y_UM / VOXEL_ISO - r_slice
        def ell = new OvalRoi((float)sx, (float)sy,
                               (float)(r_slice*2), (float)(r_slice*2))
        ell.setStrokeColor(new Color(255, 100, 100))
        ell.setStrokeWidth(2.0f)
        ov.add(ell)
    }

    // Segments
    def segList = sliceMap.get(z0, [])
    segList.each { s ->
        def roi = new Line((float)s.x1, (float)s.y1, (float)s.x2, (float)s.y2)
        roi.setStrokeWidth(LINE_WIDTH)
        roi.setStrokeColor(s.col)
        ov.add(roi)
    }

    imp.setOverlay(ov)
    imp.updateAndDraw()
}

// Draw for current slice
drawOverlay(imp.getCurrentSlice())

// ── LIVE UPDATE ON SLICE CHANGE ──────────────────────────────
def listener = [
    imageUpdated: { ImagePlus updated ->
        if (updated.is(imp)) {
            def s = imp.getCurrentSlice()
            drawOverlay(s)
        }
    },
    imageOpened:  { ImagePlus p -> },
    imageClosed:  { ImagePlus p -> },
] as ImageListener

ImagePlus.addImageListener(listener)

println ""
println "Overlay active — scroll Z slider to navigate"
println "Ctrl+Shift+O : toggle overlay"
println "Image → Adjust → Brightness/Contrast"
println ""
println "NOTE: Close the Script Editor to stop the listener"
