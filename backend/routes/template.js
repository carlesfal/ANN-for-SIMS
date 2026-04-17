'use strict';

const express = require('express');
const router = express.Router();
const ExcelJS = require('exceljs');
const { apiRateLimiter } = require('../middleware/validation');

// Colour palette for each column category
const CATEGORY_STYLES = {
  label: {
    bgArgb: 'FFD6E4FF', // light blue
    fgArgb: 'FF1E3A5F',
    prefix: 'Label',
  },
  input: {
    bgArgb: 'FFD6F5D6', // light green
    fgArgb: 'FF1A4731',
    prefix: 'Input',
  },
  output: {
    bgArgb: 'FFFFEDD6', // light amber
    fgArgb: 'FF7C2D12',
    prefix: 'Output',
  },
};

/**
 * Parse a positive integer from a query param, clamped to [min, max].
 */
function parseCount(value, min, max, defaultVal) {
  const n = parseInt(value, 10);
  if (!Number.isFinite(n)) return defaultVal;
  return Math.max(min, Math.min(max, n));
}

/**
 * GET /api/template/download
 * Query params:
 *   labels  — number of label columns  (default 1, min 0, max 20)
 *   inputs  — number of input columns  (default 3, min 1, max 50)
 *   outputs — number of output columns (default 1, min 1, max 20)
 *
 * Returns a styled .xlsx file with one header row and two example rows.
 */
router.get('/download', apiRateLimiter, async (req, res) => {
  const numLabels  = parseCount(req.query.labels,  0, 20, 1);
  const numInputs  = parseCount(req.query.inputs,  1, 50, 3);
  const numOutputs = parseCount(req.query.outputs, 1, 20, 1);

  try {
    const wb = new ExcelJS.Workbook();
    wb.creator = 'ANN-SIMS';
    wb.created = new Date();

    const ws = wb.addWorksheet('Data', {
      views: [{ state: 'frozen', ySplit: 1 }],
    });

    // Build column definitions
    const columns = [];

    for (let i = 1; i <= numLabels; i++) {
      columns.push({ category: 'label', header: `${CATEGORY_STYLES.label.prefix}_${i}`, key: `label_${i}`, width: 16 });
    }
    for (let i = 1; i <= numInputs; i++) {
      columns.push({ category: 'input', header: `${CATEGORY_STYLES.input.prefix}_${i}`, key: `input_${i}`, width: 16 });
    }
    for (let i = 1; i <= numOutputs; i++) {
      columns.push({ category: 'output', header: `${CATEGORY_STYLES.output.prefix}_${i}`, key: `output_${i}`, width: 16 });
    }

    ws.columns = columns.map(({ header, key, width }) => ({ header, key, width }));

    // Style header row
    const headerRow = ws.getRow(1);
    headerRow.height = 24;
    columns.forEach((col, idx) => {
      const cell = headerRow.getCell(idx + 1);
      const style = CATEGORY_STYLES[col.category];
      cell.fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: style.bgArgb },
      };
      cell.font = {
        bold: true,
        color: { argb: style.fgArgb },
        size: 11,
      };
      cell.alignment = { horizontal: 'center', vertical: 'middle' };
      cell.border = {
        top:    { style: 'thin', color: { argb: 'FFB0B8C1' } },
        bottom: { style: 'thin', color: { argb: 'FFB0B8C1' } },
        left:   { style: 'thin', color: { argb: 'FFB0B8C1' } },
        right:  { style: 'thin', color: { argb: 'FFB0B8C1' } },
      };
    });

    // Add two placeholder example rows
    const exampleRows = [
      columns.reduce((row, col, idx) => {
        row[col.key] = col.category === 'label' ? `Sample_A` : (idx + 1) * 1.0;
        return row;
      }, {}),
      columns.reduce((row, col, idx) => {
        row[col.key] = col.category === 'label' ? `Sample_B` : (idx + 1) * 2.0;
        return row;
      }, {}),
    ];
    ws.addRows(exampleRows);

    // Style example rows
    for (let r = 2; r <= 3; r++) {
      const row = ws.getRow(r);
      row.height = 20;
      columns.forEach((col, idx) => {
        const cell = row.getCell(idx + 1);
        cell.alignment = { horizontal: 'center', vertical: 'middle' };
        cell.font = { size: 11, italic: true, color: { argb: 'FF6B7280' } };
      });
    }

    // Add a legend sheet
    const legend = wb.addWorksheet('Legend');
    legend.columns = [
      { header: 'Category', key: 'cat', width: 14 },
      { header: 'Colour', key: 'colour', width: 14 },
      { header: 'Description', key: 'desc', width: 48 },
    ];

    const legendData = [
      { cat: 'Label', colour: 'Blue', desc: 'Identifier / sample name columns — not used as model features.' },
      { cat: 'Input', colour: 'Green', desc: 'Feature columns fed into the ANN as inputs.' },
      { cat: 'Output', colour: 'Amber', desc: 'Target column(s) the model learns to predict.' },
    ];

    const legendStyles = [CATEGORY_STYLES.label, CATEGORY_STYLES.input, CATEGORY_STYLES.output];

    const legendHeader = legend.getRow(1);
    legendHeader.height = 22;
    [1, 2, 3].forEach((c) => {
      const cell = legendHeader.getCell(c);
      cell.font = { bold: true, size: 11 };
      cell.alignment = { horizontal: 'center', vertical: 'middle' };
    });

    legendData.forEach((d, i) => {
      legend.addRow(d);
      const row = legend.getRow(i + 2);
      row.height = 20;
      const catCell = row.getCell(1);
      catCell.fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: legendStyles[i].bgArgb },
      };
      catCell.font = { bold: true, color: { argb: legendStyles[i].fgArgb }, size: 11 };
      catCell.alignment = { horizontal: 'center', vertical: 'middle' };
    });

    const filename = `ann_sims_template_L${numLabels}_I${numInputs}_O${numOutputs}.xlsx`;
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);

    await wb.xlsx.write(res);
    res.end();
  } catch (err) {
    console.error('[TEMPLATE]', err);
    res.status(500).json({ error: 'Failed to generate template', details: err.message });
  }
});

module.exports = router;
