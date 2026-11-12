# UIE themes

Every `*.json` file in this folder is a theme. `base.qss` is the shared stylesheet template.

## Add a theme
1. Copy `bloomberg_terminal.json` to `my_theme.json`.
2. Change `id` (must be unique), `name`, and the colors.
3. Restart the app, or use **View > Reload Themes**. It appears in the theme dropdown.

You can also drop themes in `~/.uie/themes/` so they survive app updates.

## Color tokens
bg, panel, panel_alt, header_bg, border, text, text_dim, accent, accent_hover, accent_text (text drawn on accent),
positive, negative, warning, info, selection, selection_text, input_bg.
