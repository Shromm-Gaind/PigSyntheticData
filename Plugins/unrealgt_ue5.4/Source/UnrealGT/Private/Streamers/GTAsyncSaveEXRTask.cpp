#include "Streamers/GTAsyncSaveEXRTask.h"

#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"
#include <cstring>
#include <string>
#include <vector>

void write_string_attr(char* buffer, const std::string& key, const std::string& value)
{
    strcat(buffer, key.c_str());
    strcat(buffer, "\0");
    strcat(buffer, "string\0");
    int32_t length = value.length() + 1;
    memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
    strcat(buffer, value.c_str());
    strcat(buffer, "\0");
}

void write_int_attr(char* buffer, const std::string& key, int32_t value)
{
    strcat(buffer, key.c_str());
    strcat(buffer, "\0");
    strcat(buffer, "int\0");
    int32_t length = sizeof(int32_t);
    memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
    memcpy(buffer + strlen(buffer), &value, sizeof(int32_t));
}

void write_box2i_attr(char* buffer, const std::string& key, const Box2i& box)
{
    strcat(buffer, key.c_str());
    strcat(buffer, "\0");
    strcat(buffer, "box2i\0");
    int32_t length = sizeof(Box2i);
    memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
    memcpy(buffer + strlen(buffer), &box, sizeof(Box2i));
}

void write_channel_attr(char* buffer, const std::string& key, const std::vector<Channel>& channels)
{
    strcat(buffer, key.c_str());
    strcat(buffer, "\0");
    strcat(buffer, "channels\0");
    int32_t length = channels.size() * sizeof(Channel) + 1;
    memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
    for (const auto& channel : channels)
    {
        strcat(buffer, channel.name);
        strcat(buffer, "\0");
        memcpy(buffer + strlen(buffer), &channel.type, sizeof(int32_t));
        memcpy(buffer + strlen(buffer), &channel.pLinear, sizeof(uint8_t));
        strcat(buffer, "\0\0\0");  // Reserved
    }
    strcat(buffer, "\0");
}

void create_exr_header(FILE* file, int width, int height)
{
    ExrHeader header{};
    header.magic_number = INTEL_ORDER32(20000630);  // OpenEXR magic number
    header.version = INTEL_ORDER32(2);              // Version number
    header.chunkCount = 1;

    header.dataWindow = {0, 0, width - 1, height - 1};
    header.displayWindow = {0, 0, width - 1, height - 1};
    header.pixelAspectRatio = 1.0;
    header.lineOrder = INTEL_ORDER32(0);  // Increasing Y

    std::vector<Channel> channels = {{"Z", INTEL_ORDER32(2), 1, {0}}};  // 'Z' channel, 32-bit float
    memset(header.channels, 0, sizeof(header.channels));
    write_channel_attr(header.channels, "channels", channels);

    memset(header.compression, 0, sizeof(header.compression));
    strcpy(header.compression, "PIZ_COMPRESSION");

    fwrite(&header, sizeof(ExrHeader), 1, file);
}

void FGTAsyncSaveEXRTask::DoWork()
{
    FILE* File = fopen(TCHAR_TO_ANSI(*FilePath), "wb");
    if (!File)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to open file for writing: %s"), *FilePath);
        return;
    }

    create_exr_header(File, Width, Height);

    // Write depth pixels (if you use a float array, update this line)
    fwrite(Image.Pixels.GetData(), sizeof(FFloat16Color), Image.Pixels.Num(), File);

    fclose(File);
}
