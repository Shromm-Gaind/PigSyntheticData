// Fill out your copyright notice in the Description page of Project Settings.

#include "Streamers/GTAsyncMakeImageTask.h"

#include "Async/Async.h"
#include "Modules/ModuleManager.h"
#include "Serialization/MemoryWriter.h"
#include <IImageWrapper.h>
#include <IImageWrapperModule.h>

#include <cstring>
#include <string>
#include <vector>

#include "Generators/Image/GTImageGeneratorBase.h"

FGTAsyncMakeImageTask::FGTAsyncMakeImageTask(
    UGTImageGeneratorBase* SourceComponent,
    const FGTImage& Image,
    EGTImageFileFormat ImageFormat,
    bool bWriteAlpha,
    FDateTime TimeStamp)
    : SourceComponent(SourceComponent)
    , Image(Image)
    , ImageFormat(ImageFormat)
    , TimeStamp(TimeStamp)
    , bWriteAlpha(bWriteAlpha)
{
    FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
}

void FGTAsyncMakeImageTask::DoWork()
{
    IImageWrapperModule& ImageWrapperModule =
        FModuleManager::GetModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    TArray<uint8> ImgData;
    switch (ImageFormat)
    {
        case EGTImageFileFormat::PNG:
        {
            TSharedPtr<IImageWrapper> ImageWrapper =
                ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
            ImageWrapper->SetRaw(
                Image.Pixels.GetData(),
                Image.Pixels.GetAllocatedSize(),
                Image.Width,
                Image.Height,
                ERGBFormat::BGRA,
                8);
            ImgData = ImageWrapper->GetCompressed(100);
            break;
        }
        case EGTImageFileFormat::BMP:
        {
            // TSharedPtr<IImageWrapper> ImageWrapper =
            // ImageWrapperModule.CreateImageWrapper(EImageFormat::BMP);
            // ImageWrapper->SetRaw(Image.Pixels.GetData(), Image.Pixels.GetAllocatedSize(),
            // Image.Width, Image.Height, ERGBFormat::BGRA, 8); ImgData =
            // ImageWrapper->GetCompressed(100);

            // bitmap support in imagewrapper is buggy
            int Width = Image.Width;
            int Height = Image.Height;

            FColor* Data = Image.Pixels.GetData();

            // TODO configure?
            FIntRect SubRectangle(0, 0, Width, Height);

            FMemoryWriter Ar(ImgData);

#if PLATFORM_SUPPORTS_PRAGMA_PACK
#pragma pack(push, 1)
#endif
            struct BITMAPFILEHEADER
            {
                uint16 bfType GCC_PACK(1);
                uint32 bfSize GCC_PACK(1);
                uint16 bfReserved1 GCC_PACK(1);
                uint16 bfReserved2 GCC_PACK(1);
                uint32 bfOffBits GCC_PACK(1);
            } FH;
            struct BITMAPINFOHEADER
            {
                uint32 biSize GCC_PACK(1);
                int32 biWidth GCC_PACK(1);
                int32 biHeight GCC_PACK(1);
                uint16 biPlanes GCC_PACK(1);
                uint16 biBitCount GCC_PACK(1);
                uint32 biCompression GCC_PACK(1);
                uint32 biSizeImage GCC_PACK(1);
                int32 biXPelsPerMeter GCC_PACK(1);
                int32 biYPelsPerMeter GCC_PACK(1);
                uint32 biClrUsed GCC_PACK(1);
                uint32 biClrImportant GCC_PACK(1);
            } IH;
            struct BITMAPV4HEADER
            {
                uint32 bV4RedMask GCC_PACK(1);
                uint32 bV4GreenMask GCC_PACK(1);
                uint32 bV4BlueMask GCC_PACK(1);
                uint32 bV4AlphaMask GCC_PACK(1);
                uint32 bV4CSType GCC_PACK(1);
                uint32 bV4EndpointR[3] GCC_PACK(1);
                uint32 bV4EndpointG[3] GCC_PACK(1);
                uint32 bV4EndpointB[3] GCC_PACK(1);
                uint32 bV4GammaRed GCC_PACK(1);
                uint32 bV4GammaGreen GCC_PACK(1);
                uint32 bV4GammaBlue GCC_PACK(1);
            } IHV4;
#if PLATFORM_SUPPORTS_PRAGMA_PACK
#pragma pack(pop)
#endif

            uint32 BytesPerPixel = bWriteAlpha ? 4 : 3;
            uint32 BytesPerLine = Align(Width * BytesPerPixel, 4);

            uint32 InfoHeaderSize =
                sizeof(BITMAPINFOHEADER) + (bWriteAlpha ? sizeof(BITMAPV4HEADER) : 0);

            // File header.
            FH.bfType = INTEL_ORDER16((uint16)('B' + 256 * 'M'));
            FH.bfSize = INTEL_ORDER32(
                (uint32)(sizeof(BITMAPFILEHEADER) + InfoHeaderSize + BytesPerLine * Height));
            FH.bfReserved1 = INTEL_ORDER16((uint16)0);
            FH.bfReserved2 = INTEL_ORDER16((uint16)0);
            FH.bfOffBits = INTEL_ORDER32((uint32)(sizeof(BITMAPFILEHEADER) + InfoHeaderSize));
            Ar.Serialize(&FH, sizeof(FH));

            // Info header.
            IH.biSize = INTEL_ORDER32((uint32)InfoHeaderSize);
            IH.biWidth = INTEL_ORDER32((uint32)Width);
            IH.biHeight = INTEL_ORDER32((uint32)Height);
            IH.biPlanes = INTEL_ORDER16((uint16)1);
            IH.biBitCount = INTEL_ORDER16((uint16)BytesPerPixel * 8);
            if (bWriteAlpha)
            {
                IH.biCompression = INTEL_ORDER32((uint32)3);  // BI_BITFIELDS
            }
            else
            {
                IH.biCompression = INTEL_ORDER32((uint32)0);  // BI_RGB
            }
            IH.biSizeImage = INTEL_ORDER32((uint32)BytesPerLine * Height);
            IH.biXPelsPerMeter = INTEL_ORDER32((uint32)0);
            IH.biYPelsPerMeter = INTEL_ORDER32((uint32)0);
            IH.biClrUsed = INTEL_ORDER32((uint32)0);
            IH.biClrImportant = INTEL_ORDER32((uint32)0);
            Ar.Serialize(&IH, sizeof(IH));

            // If we're writing alpha, we need to write the extra portion of the V4 header
            if (bWriteAlpha)
            {
                IHV4.bV4RedMask = INTEL_ORDER32((uint32)0x00ff0000);
                IHV4.bV4GreenMask = INTEL_ORDER32((uint32)0x0000ff00);
                IHV4.bV4BlueMask = INTEL_ORDER32((uint32)0x000000ff);
                IHV4.bV4AlphaMask = INTEL_ORDER32((uint32)0xff000000);
                IHV4.bV4CSType = INTEL_ORDER32((uint32)'Win ');
                IHV4.bV4GammaRed = INTEL_ORDER32((uint32)0);
                IHV4.bV4GammaGreen = INTEL_ORDER32((uint32)0);
                IHV4.bV4GammaBlue = INTEL_ORDER32((uint32)0);
                Ar.Serialize(&IHV4, sizeof(IHV4));
            }

            // Colors.
            for (int32 i = SubRectangle.Max.Y - 1; i >= SubRectangle.Min.Y; i--)
            {
                for (int32 j = SubRectangle.Min.X; j < SubRectangle.Max.X; j++)
                {
                    Ar.Serialize((void*)&Data[i * Width + j].B, 1);
                    Ar.Serialize((void*)&Data[i * Width + j].G, 1);
                    Ar.Serialize((void*)&Data[i * Width + j].R, 1);

                    if (bWriteAlpha)
                    {
                        Ar.Serialize((void*)&Data[i * Width + j].A, 1);
                    }
                }

                // Pad each row's length to be a multiple of 4 bytes.

                for (uint32 PadIndex = Width * BytesPerPixel; PadIndex < BytesPerLine; PadIndex++)
                {
                    uint8 B = 0;
                    Ar.Serialize(&B, 1);
                }
            }
            break;
        }
        case EGTImageFileFormat::EXR:
        {
            // Define the necessary structs for the EXR file format
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

            // Define the functions to handle EXR attributes
            auto write_string_attr =
                [](char* buffer, const std::string& key, const std::string& value)
            {
                strcat(buffer, key.c_str());
                strcat(buffer, "\0");
                strcat(buffer, "string\0");
                int32_t length = value.length() + 1;
                memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
                strcat(buffer, value.c_str());
                strcat(buffer, "\0");
            };

            auto write_int_attr = [](char* buffer, const std::string& key, int32_t value)
            {
                strcat(buffer, key.c_str());
                strcat(buffer, "\0");
                strcat(buffer, "int\0");
                int32_t length = sizeof(int32_t);
                memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
                memcpy(buffer + strlen(buffer), &value, sizeof(int32_t));
            };

            auto write_box2i_attr = [](char* buffer, const std::string& key, const Box2i& box)
            {
                strcat(buffer, key.c_str());
                strcat(buffer, "\0");
                strcat(buffer, "box2i\0");
                int32_t length = sizeof(Box2i);
                memcpy(buffer + strlen(buffer), &length, sizeof(int32_t));
                memcpy(buffer + strlen(buffer), &box, sizeof(Box2i));
            };

            auto write_channel_attr =
                [](char* buffer, const std::string& key, const std::vector<Channel>& channels)
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
            };

            // Create and populate the EXR header
            ExrHeader header{};
            header.magic_number = INTEL_ORDER32(20000630);  // OpenEXR magic number
            header.version = INTEL_ORDER32(2);              // Version number
            header.chunkCount = 1;

            header.dataWindow = {0, 0, Image.Width - 1, Image.Height - 1};
            header.displayWindow = {0, 0, Image.Width - 1, Image.Height - 1};
            header.pixelAspectRatio = 1.0;
            header.lineOrder = INTEL_ORDER32(0);  // Increasing Y

            std::vector<Channel> channels = {
                {"Z", INTEL_ORDER32(2), 1, {0}}};  // 'Z' channel, 32-bit float
            memset(header.channels, 0, sizeof(header.channels));
            write_channel_attr(header.channels, "channels", channels);

            memset(header.compression, 0, sizeof(header.compression));
            strcpy(header.compression, "PIZ_COMPRESSION");

            // Check if DepthPixels has data
            if (Image.Pixels.Num() == 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("No depth data found in Image.Pixels."));
                return;  // Early exit if no data
            }
            else
            {
                UE_LOG(LogTemp, Log, TEXT("Depth data found: %d pixels."), Image.Pixels.Num());
            }
            /*
                        // Log first few depth values to check content
            for (int32 i = 0; i < FMath::Min(10, Image.Pixels.Num()); ++i)
            {
                UE_LOG(LogTemp, Log, TEXT("Depth value at index %d: %f"), i, Image.DepthPixels[i]);
            }
            */


            // Serialize the EXR data into a TArray<uint8>
            TArray<uint8> ExrImgData;  // Or any other descriptive name
            FMemoryWriter Ar(ExrImgData);


            // Write the header
            Ar.Serialize(&header, sizeof(ExrHeader));

                // Check the size of the data being written
            size_t DepthDataSize = sizeof(float) * Image.Pixels.Num();
            UE_LOG(LogTemp, Log, TEXT("Writing %d bytes of depth data to EXR."), DepthDataSize);


            // Write the depth data
            Ar.Serialize(Image.Pixels.GetData(), Image.Pixels.Num() * sizeof(float));
        }
        break;
    }

    UGTImageGeneratorBase* SourceComponentLocal = SourceComponent;
    FDateTime TimeStamptLocal = TimeStamp;

    // TODO figure out if we can pass ImgData by ref
    // probably not because this thread could be dead by the time the lambda is called on the
    // gamethread
    AsyncTask(
        ENamedThreads::GameThread,
        [SourceComponentLocal, ImgData, TimeStamptLocal]()
        {
            if (SourceComponentLocal && SourceComponentLocal->IsValidLowLevelFast())
            {
                SourceComponentLocal->DataReadyDelegate.Broadcast(ImgData, TimeStamptLocal);
            }
        });
}

TStatId FGTAsyncMakeImageTask::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(FGTMakeImageTask, STATGROUP_ThreadPoolAsyncTasks);
}