/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/editors/space_collections/collections_draw.c
 *  \ingroup spcollections
 */

#include <string.h>

#include "BLF_api.h"

#include "BLI_rect.h"

#include "DNA_screen_types.h"
#include "DNA_space_types.h"

#include "UI_resources.h"
#include "UI_table.h"

#include "collections_intern.h"


void collections_draw_table(SpaceCollections *spc, const ARegion *ar)
{
	UI_table_max_width_set(spc->table, BLI_rctf_size_x(&ar->v2d.tot));
	UI_table_draw(spc->table);
}

void collections_draw_cell(void *rowdata, rcti drawrect)
{
	LayerCollection *collection = rowdata;
	const char *name = collection->scene_collection->name;

	UI_ThemeColor(TH_TEXT);
	BLF_draw_default(drawrect.xmin, drawrect.ymin, 0.0f, name, strlen(name));
}
