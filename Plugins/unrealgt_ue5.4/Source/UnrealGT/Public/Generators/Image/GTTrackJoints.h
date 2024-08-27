
#pragma once

#include "Components/ActorComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "CoreMinimal.h"

#include "GTImageGeneratorBase.h"
#include "GTImage.h"
#include "GTTrackJoints.generated.h"

class UGTSceneCaptureComponent2D;


UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class UNREALGT_API UGTTrackJoints : public UGTDataGeneratorComponent
{
    GENERATED_BODY()

public:
    UGTTrackJoints();

    void GenerateData(const FDateTime& TimeStamp);


protected:
    virtual void BeginPlay() override;

public:
    void UpdateSocketLocations(USkeletalMeshComponent* SkeletalMeshComponent);

private:
    TArray<FVector> SocketLocations;

    USkeletalMeshComponent* FoundSkeletalMeshComponent = nullptr;

    UPROPERTY()
    UGTSceneCaptureComponent2D* SceneCaptureComponent;
};

