/*
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
 */

#ifndef __BKE_ASSET_H__
#define __BKE_ASSET_H__

/** \file
 * \ingroup bke
 */

#include "BLI_utildefines.h"

#ifdef __cplusplus
extern "C" {
#endif

struct BlendWriter;
struct BlendDataReader;
struct ID;
struct PreviewImage;

#ifdef WITH_ASSET_REPO_INFO
struct AssetRepositoryInfo *BKE_asset_repository_info_global_ensure(void);
void BKE_asset_repository_info_free(struct AssetRepositoryInfo **repository_info);
void BKE_asset_repository_info_global_free(void);
void BKE_asset_repository_info_update_for_file_read(struct AssetRepositoryInfo **repository_info);
#endif

struct AssetCatalog *BKE_asset_repository_catalog_create(const char *name);
void BKE_asset_repository_catalog_free(struct AssetCatalog *catalog);

struct AssetData *BKE_asset_data_create(void);
void BKE_asset_data_free(struct AssetData *asset_data);

struct CustomTagEnsureResult {
  struct CustomTag *tag;
  /* Set to false if a tag of this name was already present. */
  bool is_new;
};

struct CustomTag *BKE_assetdata_tag_add(struct AssetData *asset_data, const char *name);
struct CustomTagEnsureResult BKE_assetdata_tag_ensure(struct AssetData *asset_data,
                                                      const char *name);
void BKE_assetdata_tag_remove(struct AssetData *asset_data, struct CustomTag *tag);

struct PreviewImage *BKE_assetdata_preview_get_from_id(const struct AssetData *asset_data,
                                                       const struct ID *owner_id);

void BKE_assetdata_write(struct BlendWriter *writer, struct AssetData *asset_data);
void BKE_assetdata_read(struct BlendDataReader *reader, struct AssetData *asset_data);

#ifdef __cplusplus
}
#endif

#endif /* __BKE_ASSET_H__ */