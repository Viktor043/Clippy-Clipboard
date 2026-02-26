import Foundation
import CoreData
import AppKit

final class ClipboardStore: ObservableObject {
    static let shared = ClipboardStore()

    @Published var items: [ClipboardItem] = []

    private let container: NSPersistentContainer

    /// Dynamic max items — reads from UserDefaults so settings changes take effect immediately
    private var maxItems: Int {
        let stored = UserDefaults.standard.integer(forKey: "maxHistoryItems")
        return stored > 0 ? stored : AppConstants.maxHistoryItems
    }

    init() {
        let model = ClipboardStore.createModel()
        container = NSPersistentContainer(name: "ClipboardModel", managedObjectModel: model)
        container.loadPersistentStores { _, error in
            if let error = error {
                print("Core Data failed to load: \(error.localizedDescription)")
            }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
        fetchItems()
    }

    // MARK: - Programmatic Core Data Model

    private static func createModel() -> NSManagedObjectModel {
        let model = NSManagedObjectModel()

        let entity = NSEntityDescription()
        entity.name = "CDClipboardItem"
        entity.managedObjectClassName = NSStringFromClass(NSManagedObject.self)

        let idAttr = NSAttributeDescription()
        idAttr.name = "id"
        idAttr.attributeType = .UUIDAttributeType
        idAttr.isOptional = false

        let typeAttr = NSAttributeDescription()
        typeAttr.name = "type"
        typeAttr.attributeType = .stringAttributeType
        typeAttr.isOptional = false

        let timestampAttr = NSAttributeDescription()
        timestampAttr.name = "timestamp"
        timestampAttr.attributeType = .dateAttributeType
        timestampAttr.isOptional = false

        let textAttr = NSAttributeDescription()
        textAttr.name = "textContent"
        textAttr.attributeType = .stringAttributeType
        textAttr.isOptional = true

        let imageAttr = NSAttributeDescription()
        imageAttr.name = "imageData"
        imageAttr.attributeType = .binaryDataAttributeType
        imageAttr.isOptional = true
        imageAttr.allowsExternalBinaryDataStorage = true

        let thumbAttr = NSAttributeDescription()
        thumbAttr.name = "thumbnailData"
        thumbAttr.attributeType = .binaryDataAttributeType
        thumbAttr.isOptional = true

        let rawAttr = NSAttributeDescription()
        rawAttr.name = "rawData"
        rawAttr.attributeType = .binaryDataAttributeType
        rawAttr.isOptional = true
        rawAttr.allowsExternalBinaryDataStorage = true

        let sourceAttr = NSAttributeDescription()
        sourceAttr.name = "sourceAppBundleID"
        sourceAttr.attributeType = .stringAttributeType
        sourceAttr.isOptional = true

        let typesAttr = NSAttributeDescription()
        typesAttr.name = "pasteboardTypes"
        typesAttr.attributeType = .transformableAttributeType
        typesAttr.isOptional = true
        typesAttr.valueTransformerName = "NSSecureUnarchiveFromData"

        entity.properties = [
            idAttr, typeAttr, timestampAttr, textAttr,
            imageAttr, thumbAttr, rawAttr, sourceAttr, typesAttr
        ]

        model.entities = [entity]
        return model
    }

    // MARK: - CRUD

    func addItem(_ item: ClipboardItem) {
        let context = container.viewContext

        // Deduplicate files: if a file with the same path already exists, remove the old entry
        // so it gets "bumped" to the top instead of creating a duplicate
        if item.type == .file, let path = item.textContent {
            removeDuplicateFile(path: path, in: context)
        }

        let entity = NSEntityDescription.entity(forEntityName: "CDClipboardItem", in: context)!
        let obj = NSManagedObject(entity: entity, insertInto: context)

        obj.setValue(item.id, forKey: "id")
        obj.setValue(item.type.rawValue, forKey: "type")
        obj.setValue(item.timestamp, forKey: "timestamp")
        obj.setValue(item.textContent, forKey: "textContent")
        obj.setValue(item.imageData, forKey: "imageData")
        obj.setValue(item.thumbnailData, forKey: "thumbnailData")
        obj.setValue(item.rawData, forKey: "rawData")
        obj.setValue(item.sourceAppBundleID, forKey: "sourceAppBundleID")
        obj.setValue(item.pasteboardTypes as NSArray, forKey: "pasteboardTypes")

        save()
        pruneOldItems()
        fetchItems()
    }

    func deleteItem(id: UUID) {
        let context = container.viewContext
        let request = NSFetchRequest<NSManagedObject>(entityName: "CDClipboardItem")
        request.predicate = NSPredicate(format: "id == %@", id as CVarArg)

        if let results = try? context.fetch(request), let obj = results.first {
            context.delete(obj)
            save()
            fetchItems()
        }
    }

    func clearAll() {
        let context = container.viewContext
        let request = NSFetchRequest<NSManagedObject>(entityName: "CDClipboardItem")

        if let results = try? context.fetch(request) {
            for obj in results {
                context.delete(obj)
            }
            save()
            fetchItems()
        }
    }

    func fetchItems() {
        let context = container.viewContext
        let request = NSFetchRequest<NSManagedObject>(entityName: "CDClipboardItem")
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        request.fetchLimit = maxItems

        guard let results = try? context.fetch(request) else {
            items = []
            return
        }

        items = results.compactMap { obj in
            guard let id = obj.value(forKey: "id") as? UUID,
                  let typeStr = obj.value(forKey: "type") as? String,
                  let type = ClipboardItemType(rawValue: typeStr),
                  let timestamp = obj.value(forKey: "timestamp") as? Date else {
                return nil
            }

            return ClipboardItem(
                id: id,
                type: type,
                timestamp: timestamp,
                textContent: obj.value(forKey: "textContent") as? String,
                imageData: obj.value(forKey: "imageData") as? Data,
                thumbnailData: obj.value(forKey: "thumbnailData") as? Data,
                rawData: obj.value(forKey: "rawData") as? Data,
                sourceAppBundleID: obj.value(forKey: "sourceAppBundleID") as? String,
                pasteboardTypes: (obj.value(forKey: "pasteboardTypes") as? [String]) ?? []
            )
        }
    }

    // MARK: - Private

    /// Remove any existing file entries with the same path to prevent duplicates
    private func removeDuplicateFile(path: String, in context: NSManagedObjectContext) {
        let request = NSFetchRequest<NSManagedObject>(entityName: "CDClipboardItem")
        request.predicate = NSPredicate(format: "type == %@ AND textContent == %@", "file", path)

        if let duplicates = try? context.fetch(request) {
            for obj in duplicates {
                context.delete(obj)
            }
        }
    }

    private func pruneOldItems() {
        let context = container.viewContext
        let request = NSFetchRequest<NSManagedObject>(entityName: "CDClipboardItem")
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        let limit = maxItems

        guard let results = try? context.fetch(request), results.count > limit else { return }

        for obj in results.dropFirst(limit) {
            context.delete(obj)
        }
        save()
    }

    private func save() {
        let context = container.viewContext
        guard context.hasChanges else { return }
        do {
            try context.save()
        } catch {
            print("Core Data save error: \(error.localizedDescription)")
        }
    }
}
