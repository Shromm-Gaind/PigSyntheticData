#pragma once

#include "Async/AsyncWork.h"
#include "CoreMinimal.h"
#include <string>
#include <vector>

#include "GTImage.h"

#define INTEL_ORDER32(x) (x)

struct Box2i
{
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};

struct Channel
{
    char name[32];
    int32_t type;
    uint8_t pLinear;
    uint8_t reserved[3];
};

struct ExrHeader
{
    uint32_t magic_number;
    uint32_t version;
    uint32_t chunkCount;
    Box2i dataWindow;
    Box2i displayWindow;
    float pixelAspectRatio;
    int32_t lineOrder;
    char channels[64];
    char compression[32];
};
void write_string_attr(char* buffer, const std::string& key, const std::string& value);
void write_int_attr(char* buffer, const std::string& key, int32_t value);
void write_box2i_attr(char* buffer, const std::string& key, const Box2i& box);
void write_channel_attr(char* buffer, const std::string& key, const std::vector<Channel>& channels);
void create_exr_header(FILE* file, int width, int height);

class FGTAsyncSaveEXRTask : public FNonAbandonableTask
{
    friend class FAutoDeleteAsyncTask<FGTAsyncSaveEXRTask>;

public:
    FGTAsyncSaveEXRTask(
        const FGTImage& InImage,
        int32 InWidth,
        int32 InHeight,
        const FDateTime& InTimeStamp,
        const FString& InFilePath)
        : Image(InImage)
        , Width(InWidth)
        , Height(InHeight)
        , TimeStamp(InTimeStamp)
        , FilePath(InFilePath)
    {
    }

    void DoWork();

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FGTAsyncSaveEXRTask, STATGROUP_ThreadPoolAsyncTasks);
    }


private:
    FGTImage Image;
    int32 Width;
    int32 Height;
    FDateTime TimeStamp;
    FString FilePath;
};
